"""Run annotation-free diffusion-policy inference on official nuScenes test data."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from diffusion_policy.conditioning import DrivingConditionEncoder
from diffusion_policy.config import ACTION_DIM, DT, DatasetConfig, FUT_LEN
from diffusion_policy.dataset import (
    build_test_conditioning_dataset,
    collate_diffusion_batch,
    validate_nuscenes_dataroot,
)
from diffusion_policy.model import DrivingDiffusionPolicy
from diffusion_policy.normalization import DiffusionNormalizer
from diffusion_policy.scheduler import DrivingDiffusionScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diffusion-policy inference on nuScenes test.")
    parser.add_argument("--test-dataroot", required=True, type=Path)
    parser.add_argument("--test-version", default="v1.0-test")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("/kaggle/working/diffusion_test_outputs"), type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-inference-steps", default=100, type=int)
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(choice)


def print_environment(device: torch.device) -> None:
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"device: {device}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def normalize_condition_batch(batch: dict[str, Any], normalizer: DiffusionNormalizer, device: torch.device) -> dict[str, Any]:
    normalized = dict(batch)
    for field in ("history_speed", "history_curvature", "ego_state"):
        normalized[field] = normalizer.transform(batch[field], field=field).to(device)
    for field in ("current_position", "current_velocity", "current_curvature"):
        normalized[field] = batch[field].to(device)
    return normalized


def integrate_trajectory(
    current_position: torch.Tensor,
    current_velocity: torch.Tensor,
    predicted_speed: list[float],
    predicted_curvature: list[float],
) -> list[list[float]]:
    position = current_position.detach().cpu().double().numpy().copy()
    velocity = current_velocity.detach().cpu().double().numpy()
    yaw = math.atan2(float(velocity[1]), float(velocity[0])) if velocity.shape[0] >= 2 else 0.0
    trajectory = []
    for speed_value, curvature_value in zip(predicted_speed, predicted_curvature):
        speed = float(speed_value)
        curvature = float(curvature_value)
        distance = speed * DT
        yaw += curvature * distance
        position[0] += distance * math.cos(yaw)
        position[1] += distance * math.sin(yaw)
        trajectory.append([float(position[0]), float(position[1]), float(position[2])])
    return trajectory


@torch.no_grad()
def sample_actions(
    condition: torch.Tensor,
    diffusion_policy: DrivingDiffusionPolicy,
    scheduler: DrivingDiffusionScheduler,
    device: torch.device,
    num_inference_steps: int,
) -> torch.Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    sample = torch.randn(condition.shape[0], FUT_LEN, ACTION_DIM, device=device)
    try:
        scheduler.scheduler.set_timesteps(num_inference_steps, device=device)
    except TypeError:
        scheduler.scheduler.set_timesteps(num_inference_steps)
    for timestep in scheduler.scheduler.timesteps:
        timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        timestep_batch = torch.full(
            (condition.shape[0],),
            timestep_value,
            device=device,
            dtype=torch.long,
        )
        predicted_noise = diffusion_policy(sample, timestep_batch, condition)
        sample = scheduler.scheduler.step(predicted_noise, timestep, sample).prev_sample
    return sample


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    validate_nuscenes_dataroot(args.test_dataroot, args.test_version)
    device = resolve_device(args.device)
    print_environment(device)
    print(f"Test dataset root: {args.test_dataroot}")
    print(f"Test dataset version: {args.test_version}")
    print("No supervised loss will be computed because official nuScenes test has no labels.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    scheduler_config = checkpoint.get("scheduler_config", {})
    scheduler = DrivingDiffusionScheduler(
        num_train_timesteps=int(scheduler_config.get("num_train_timesteps", 1000)),
        beta_schedule=str(scheduler_config.get("beta_schedule", "squaredcos_cap_v2")),
        prediction_type=str(scheduler_config.get("prediction_type", "epsilon")),
    )
    condition_encoder = DrivingConditionEncoder().to(device)
    diffusion_policy = DrivingDiffusionPolicy().to(device)
    condition_encoder.load_state_dict(checkpoint["condition_encoder_state_dict"])
    diffusion_policy.load_state_dict(checkpoint["diffusion_policy_state_dict"])
    condition_encoder.eval()
    diffusion_policy.eval()

    stats_path = args.checkpoint.parent.parent / "normalization_stats.json"
    if not stats_path.is_file():
        stats_path = Path(checkpoint.get("normalization_stats_path", stats_path))
    if not stats_path.is_file():
        raise FileNotFoundError(f"normalization stats not found near checkpoint: {stats_path}")
    normalizer = DiffusionNormalizer.load(stats_path)

    dataset_config = DatasetConfig(dataroot=args.test_dataroot, version=args.test_version)
    test_dataset, scene_tokens = build_test_conditioning_dataset(dataset_config)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_diffusion_batch,
    )

    predictions = []
    for raw_batch in loader:
        batch = normalize_condition_batch(raw_batch, normalizer, device)
        condition = condition_encoder(
            history_speed=batch["history_speed"],
            history_curvature=batch["history_curvature"],
            ego_state=batch["ego_state"],
        )
        normalized_actions = sample_actions(
            condition,
            diffusion_policy,
            scheduler,
            device,
            args.num_inference_steps,
        )
        actions = normalizer.inverse_transform(normalized_actions.cpu(), field="target_actions")
        for index in range(actions.shape[0]):
            sample_actions_np = actions[index].detach().cpu()
            predicted_speed = sample_actions_np[:, 0].tolist()
            predicted_curvature = sample_actions_np[:, 1].tolist()
            trajectory = integrate_trajectory(
                raw_batch["current_position"][index],
                raw_batch["current_velocity"][index],
                predicted_speed,
                predicted_curvature,
            )
            predictions.append(
                {
                    "scene_token": raw_batch["scene_token"][index],
                    "scene_name": raw_batch["scene_name"][index],
                    "sample_token": raw_batch["sample_token"][index],
                    "predicted_speed": predicted_speed,
                    "predicted_curvature": predicted_curvature,
                    "predicted_trajectory": trajectory,
                }
            )

    predictions_dir = args.output_dir / "predictions"
    save_json(predictions_dir / "test_predictions.json", predictions)
    save_json(args.output_dir / "test_predictions.json", predictions)
    summary = {
        "test_dataroot": str(args.test_dataroot),
        "test_version": args.test_version,
        "checkpoint": str(args.checkpoint),
        "num_scenes": len(scene_tokens),
        "num_predictions": len(predictions),
        "skipped_reasons": test_dataset.skipped_reasons,
        "num_inference_steps": args.num_inference_steps,
        "supervised_loss": None,
    }
    save_json(args.output_dir / "test_summary.json", summary)
    print(f"Processed test scenes: {len(scene_tokens)}")
    print(f"Saved test predictions: {len(predictions)}")
    print(f"Skipped samples/scenes: {test_dataset.skipped_reasons or 'none'}")
    print(f"Predictions file: {predictions_dir / 'test_predictions.json'}")
    print(f"Summary file: {args.output_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
