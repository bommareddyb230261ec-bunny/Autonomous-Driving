"""Kaggle entry point for training the diffusion-policy model."""

from __future__ import annotations

import argparse
import json
import math
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffusion_policy.conditioning import DrivingConditionEncoder
from diffusion_policy.config import ACTION_DIM, CONDITION_DIM, DatasetConfig, FUT_LEN
from diffusion_policy.dataset import build_train_val_datasets, collate_diffusion_batch
from diffusion_policy.model import DrivingDiffusionPolicy
from diffusion_policy.normalization import DiffusionNormalizer, NormalizedDiffusionDataset
from diffusion_policy.scheduler import DrivingDiffusionScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OpenEMMA diffusion policy on Kaggle.")
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output-dir", default=Path("/kaggle/working/diffusion_outputs"), type=Path)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=4, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", default=None, type=Path)
    parser.add_argument("--sanity-test", action="store_true")
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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"GPU name: {gpu_name}")
    print(f"device: {device}")


def make_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, results_dir


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def gpu_memory(prefix: str) -> None:
    if not torch.cuda.is_available():
        return
    print(
        f"{prefix} GPU memory | name={torch.cuda.get_device_name(0)} "
        f"allocated={torch.cuda.memory_allocated() / 1024**2:.2f} MB "
        f"reserved={torch.cuda.memory_reserved() / 1024**2:.2f} MB "
        f"peak={torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )


def amp_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("history_speed", "history_curvature", "ego_state", "target_actions"):
        moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def assert_finite_tensor(name: str, tensor: torch.Tensor, epoch: int, batch_index: int) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} became NaN/Inf at epoch {epoch}, batch {batch_index}")


def check_gradients(modules: list[torch.nn.Module], epoch: int, batch_index: int) -> None:
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"gradient {name} became NaN/Inf at epoch {epoch}, batch {batch_index}"
                )


def batch_loss(
    batch: dict[str, Any],
    condition_encoder: DrivingConditionEncoder,
    diffusion_policy: DrivingDiffusionPolicy,
    scheduler: DrivingDiffusionScheduler,
    device: torch.device,
    epoch: int,
    batch_index: int,
) -> torch.Tensor:
    clean_actions = batch["target_actions"]
    noise = torch.randn_like(clean_actions)
    timesteps = torch.randint(
        0,
        scheduler.num_train_timesteps,
        (clean_actions.shape[0],),
        device=device,
    )
    condition = condition_encoder(
        history_speed=batch["history_speed"],
        history_curvature=batch["history_curvature"],
        ego_state=batch["ego_state"],
    )
    noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
    predicted_noise = diffusion_policy(noisy_actions, timesteps, condition)
    assert_finite_tensor("predicted_noise", predicted_noise, epoch, batch_index)
    loss = F.mse_loss(predicted_noise, noise)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"loss became NaN/Inf at epoch {epoch}, batch {batch_index}: {loss.item()}")
    return loss


def train_one_epoch(
    loader: DataLoader,
    condition_encoder: DrivingConditionEncoder,
    diffusion_policy: DrivingDiffusionPolicy,
    scheduler: DrivingDiffusionScheduler,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    max_optimizer_steps: int | None = None,
) -> float:
    condition_encoder.train()
    diffusion_policy.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    sample_count = 0
    pending_steps = 0
    optimizer_steps = 0
    total_batches = len(loader)
    max_batches = total_batches
    if max_optimizer_steps is not None:
        max_batches = min(total_batches, max_optimizer_steps * accumulation_steps)

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        batch_size = batch["target_actions"].shape[0]
        with amp_context(device):
            loss = batch_loss(
                batch,
                condition_encoder,
                diffusion_policy,
                scheduler,
                device,
                epoch,
                batch_index,
            )
        total_loss += loss.detach().item() * batch_size
        sample_count += batch_size
        pending_steps += 1
        segment_start = (batch_index // accumulation_steps) * accumulation_steps
        segment_size = min(accumulation_steps, max_batches - segment_start)
        is_boundary = pending_steps == segment_size
        scaled_loss = loss / segment_size

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if is_boundary:
            if scaler is not None:
                scaler.unscale_(optimizer)
            check_gradients([condition_encoder, diffusion_policy], epoch, batch_index)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending_steps = 0

    return total_loss / max(sample_count, 1)


@torch.no_grad()
def validate(
    loader: DataLoader,
    condition_encoder: DrivingConditionEncoder,
    diffusion_policy: DrivingDiffusionPolicy,
    scheduler: DrivingDiffusionScheduler,
    device: torch.device,
    epoch: int,
) -> float:
    condition_encoder.eval()
    diffusion_policy.eval()
    total_loss = 0.0
    sample_count = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        with amp_context(device):
            loss = batch_loss(
                batch,
                condition_encoder,
                diffusion_policy,
                scheduler,
                device,
                epoch,
                batch_index,
            )
        batch_size = batch["target_actions"].shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
    return total_loss / max(sample_count, 1)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE noise prediction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def checkpoint_payload(
    epoch: int,
    condition_encoder: DrivingConditionEncoder,
    diffusion_policy: DrivingDiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "condition_encoder_state_dict": condition_encoder.state_dict(),
        "diffusion_policy_state_dict": diffusion_policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
        "normalization_stats_path": str(Path(config["output_dir"]) / "normalization_stats.json"),
    }


def main() -> None:
    args = parse_args()
    if not args.dataroot.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataroot}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs, batch-size, and gradient-accumulation-steps must be positive")

    device = resolve_device(args.device)
    print_environment(device)
    print(f"Dataset root: {args.dataroot}")
    print(f"Dataset version: {args.version}")
    if args.sanity_test:
        print("Sanity test mode: using a small subset and limited optimizer steps")
    elif device.type == "cpu":
        raise RuntimeError("Full training on CPU is disabled. Use --sanity-test for smoke testing.")

    checkpoints_dir, results_dir = make_output_dirs(args.output_dir)
    config_payload = {
        "dataroot": str(args.dataroot),
        "version": args.version,
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "device": str(device),
        "fut_len": FUT_LEN,
        "action_dim": ACTION_DIM,
        "condition_dim": CONDITION_DIM,
    }
    save_json(args.output_dir / "config.json", config_payload)

    dataset_config = DatasetConfig(
        dataroot=args.dataroot,
        version=args.version,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_dataset, val_dataset, train_scenes, val_scenes = build_train_val_datasets(dataset_config)
    print(f"train scenes: {len(train_scenes)}")
    print(f"validation scenes: {len(val_scenes)}")
    print(f"train samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")
    if not set(train_scenes).isdisjoint(set(val_scenes)):
        raise RuntimeError("train/validation scene overlap detected")
    print("Scene split verified: validation data was not used to fit normalization statistics")

    if args.sanity_test:
        train_dataset = Subset(train_dataset, range(min(20, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(20, len(val_dataset))))
        args.epochs = min(args.epochs, 1)

    normalizer = DiffusionNormalizer().fit(train_dataset)
    normalizer.save(args.output_dir / "normalization_stats.json", dataset_config.obs_len, dataset_config.fut_len)
    normalized_train = NormalizedDiffusionDataset(train_dataset, normalizer)
    normalized_val = NormalizedDiffusionDataset(val_dataset, normalizer)

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_diffusion_batch,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(normalized_train, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(normalized_val, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    condition_encoder = DrivingConditionEncoder().to(device)
    diffusion_policy = DrivingDiffusionPolicy().to(device)
    scheduler = DrivingDiffusionScheduler()
    optimizer = torch.optim.AdamW(
        list(condition_encoder.parameters()) + list(diffusion_policy.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device)
    print(f"ConditionEncoder trainable parameters: {parameter_count(condition_encoder)}")
    print(f"DiffusionPolicy trainable parameters: {parameter_count(diffusion_policy)}")
    print(f"Total trainable parameters: {parameter_count(condition_encoder) + parameter_count(diffusion_policy)}")

    start_epoch = 1
    best_val_loss = math.inf
    history: list[dict[str, float]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        condition_encoder.load_state_dict(checkpoint["condition_encoder_state_dict"])
        diffusion_policy.load_state_dict(checkpoint["diffusion_policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from {args.resume}; next epoch: {start_epoch}; best_val_loss: {best_val_loss}")

    for epoch in range(start_epoch, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gpu_memory(f"epoch {epoch} start")
        train_loss = train_one_epoch(
            train_loader,
            condition_encoder,
            diffusion_policy,
            scheduler,
            optimizer,
            scaler,
            device,
            epoch,
            args.gradient_accumulation_steps,
            max_optimizer_steps=3 if args.sanity_test else None,
        )
        val_loss = validate(val_loader, condition_encoder, diffusion_policy, scheduler, device, epoch)
        gpu_memory(f"epoch {epoch} end")
        learning_rate = optimizer.param_groups[0]["lr"]
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": learning_rate}
        history.append(entry)
        print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={learning_rate:g}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = checkpoint_payload(
                epoch,
                condition_encoder,
                diffusion_policy,
                optimizer,
                best_val_loss,
                config_payload,
            )
            torch.save(payload, checkpoints_dir / "best.pt")
        payload = checkpoint_payload(
            epoch,
            condition_encoder,
            diffusion_policy,
            optimizer,
            best_val_loss,
            config_payload,
        )
        torch.save(payload, checkpoints_dir / "latest.pt")

        save_json(results_dir / "training_history.json", history)
        save_loss_curve(history, results_dir / "loss_curve.png")

    summary = {
        "best_val_loss": best_val_loss,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "condition_encoder_parameters": parameter_count(condition_encoder),
        "diffusion_policy_parameters": parameter_count(diffusion_policy),
        "total_parameters": parameter_count(condition_encoder) + parameter_count(diffusion_policy),
        "latest_checkpoint": str(checkpoints_dir / "latest.pt"),
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
    }
    save_json(results_dir / "training_summary.json", summary)
    print(f"Training complete. Best checkpoint: {checkpoints_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
