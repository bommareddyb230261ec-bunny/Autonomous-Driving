"""PyTorch Dataset construction for nuScenes diffusion-policy targets."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from nuscenes import NuScenes
from torch.utils.data import Dataset

from diffusion_policy.config import DatasetConfig
from diffusion_policy.targets import get_future_speed_curvature
from utils import EstimateCurvatureFromTrajectory


@dataclass(frozen=True)
class _Window:
    scene_token: str
    scene_name: str
    sample_token: str
    current_image: str
    current_position: np.ndarray
    current_velocity: np.ndarray
    current_curvature: float
    history_speed: np.ndarray
    history_curvature: np.ndarray
    target_actions: np.ndarray
    semantic_metadata: Mapping[str, Any]


def _load_semantic_cache(cache_path: Path | None) -> dict[str, Mapping[str, Any]]:
    if cache_path is None:
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as file:
            cache = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load semantic cache: {cache_path}") from exc
    if not isinstance(cache, dict):
        raise ValueError("semantic cache must contain a JSON object")
    return {
        str(key): value
        for key, value in cache.items()
        if isinstance(value, dict)
    }


def _metadata_for_sample(
    cache: Mapping[str, Mapping[str, Any]], sample_token: str, image_path: str
) -> Mapping[str, Any]:
    return cache.get(sample_token, cache.get(image_path, {}))


def _record_skip(skipped: dict[str, int], reason: str) -> None:
    skipped[reason] = skipped.get(reason, 0) + 1


def _scene_records(nusc: NuScenes, scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
        records.append(
            {
                "sample_token": sample_token,
                "image_path": str(Path(nusc.dataroot) / camera_data["filename"]),
                "position": np.asarray(pose["translation"][:3], dtype=float),
            }
        )
        sample_token = sample["next"]
    return records


class NuScenesDiffusionDataset(Dataset):
    """Validated nuScenes windows returning raw conditioning and target tensors."""

    def __init__(self, windows: list[_Window], skipped_reasons: Mapping[str, int] | None = None):
        self._windows = windows
        self.skipped_reasons = dict(skipped_reasons or {})

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self._windows[index]
        metadata = dict(window.semantic_metadata)
        driving_intent = metadata.get("driving_intent", "")
        if not isinstance(driving_intent, str):
            driving_intent = str(driving_intent)
        detections = metadata.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        return {
            "history_speed": torch.from_numpy(window.history_speed.copy()).float(),
            "history_curvature": torch.from_numpy(window.history_curvature.copy()).float(),
            "ego_state": torch.from_numpy(
                np.concatenate(
                    (
                        window.current_position,
                        window.current_velocity,
                        np.asarray([window.current_curvature], dtype=float),
                    )
                )
            ).float(),
            "driving_intent": driving_intent,
            "detections": detections,
            "target_actions": torch.from_numpy(window.target_actions.copy()).float(),
            "current_image": window.current_image,
            "sample_token": window.sample_token,
            "scene_token": window.scene_token,
            "scene_name": window.scene_name,
        }


def collate_diffusion_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack numerical fields while preserving variable-length raw metadata."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    tensor_keys = {"history_speed", "history_curvature", "ego_state", "target_actions"}
    return {
        key: torch.stack([sample[key] for sample in samples])
        if key in tensor_keys
        else [sample[key] for sample in samples]
        for key in samples[0]
    }


def build_train_val_datasets(
    config: DatasetConfig,
) -> tuple[NuScenesDiffusionDataset, NuScenesDiffusionDataset, set[str], set[str]]:
    """Build deterministic scene-disjoint train and validation datasets."""
    if not 0.0 < config.train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")
    if config.obs_len <= 0 or config.fut_len <= 0:
        raise ValueError("obs_len and fut_len must be positive")

    semantic_cache = _load_semantic_cache(config.semantic_cache)
    nusc = NuScenes(version=config.version, dataroot=str(config.dataroot), verbose=False)
    scenes = list(nusc.scene)
    rng = np.random.default_rng(config.seed)
    scene_order = rng.permutation(len(scenes))
    train_count = int(len(scenes) * config.train_split)
    if len(scenes) > 1:
        train_count = min(max(train_count, 1), len(scenes) - 1)
    train_indices = set(int(index) for index in scene_order[:train_count])
    train_scene_tokens = {scenes[index]["token"] for index in train_indices}
    val_scene_tokens = {
        scene["token"] for index, scene in enumerate(scenes) if index not in train_indices
    }

    all_windows: dict[str, list[_Window]] = {"train": [], "validation": []}
    skipped: dict[str, int] = {}
    total_scene_samples = 0
    for scene_index, scene in enumerate(scenes):
        records = _scene_records(nusc, scene)
        total_scene_samples += len(records)
        positions = np.asarray([record["position"] for record in records], dtype=float)
        if len(records) < config.obs_len + config.fut_len:
            _record_skip(skipped, "scene_too_short")
            continue
        if not np.isfinite(positions).all():
            _record_skip(skipped, "non_finite_scene_pose")
            continue

        velocities = np.zeros_like(positions)
        velocities[1:] = np.diff(positions, axis=0) / config.dt
        velocities[0] = velocities[1]
        try:
            curvatures = EstimateCurvatureFromTrajectory(positions)
        except (IndexError, ValueError, FloatingPointError):
            _record_skip(skipped, "curvature_estimation_failure")
            continue
        speeds = np.linalg.norm(velocities, axis=1)
        split = "train" if scene_index in train_indices else "validation"
        for start in range(len(records) - config.obs_len - config.fut_len + 1):
            current_index = start + config.obs_len - 1
            future_start = current_index + 1
            future_end = future_start + config.fut_len
            try:
                record = records[current_index]
                if not Path(record["image_path"]).is_file():
                    raise ValueError("current_image_missing")
                target_actions = get_future_speed_curvature(
                    positions[current_index],
                    positions[future_start:future_end],
                    config.dt,
                    config.fut_len,
                )
                history_speed = speeds[start : start + config.obs_len]
                history_curvature = curvatures[start : start + config.obs_len]
                current_velocity = velocities[current_index]
                current_curvature = curvatures[current_index]
                arrays = (history_speed, history_curvature, current_velocity, target_actions)
                if not all(np.isfinite(array).all() for array in arrays):
                    raise ValueError("sample contains NaN or Inf values")
            except ValueError as exc:
                _record_skip(skipped, str(exc))
                continue
            all_windows[split].append(
                _Window(
                    scene_token=scene["token"],
                    scene_name=scene["name"],
                    sample_token=record["sample_token"],
                    current_image=record["image_path"],
                    current_position=positions[current_index],
                    current_velocity=current_velocity,
                    current_curvature=float(current_curvature),
                    history_speed=history_speed,
                    history_curvature=history_curvature,
                    target_actions=target_actions,
                    semantic_metadata=_metadata_for_sample(
                        semantic_cache, record["sample_token"], record["image_path"]
                    ),
                )
            )

    print(f"Number of scenes: {len(scenes)}")
    print(f"Number of scene samples: {total_scene_samples}")
    print(f"Number of samples: {len(all_windows['train']) + len(all_windows['validation'])}")
    print(f"Number of training samples: {len(all_windows['train'])}")
    print(f"Number of validation samples: {len(all_windows['validation'])}")
    print(f"Skipped samples/scenes by reason: {skipped or 'none'}")
    return (
        NuScenesDiffusionDataset(all_windows["train"], skipped),
        NuScenesDiffusionDataset(all_windows["validation"], skipped),
        train_scene_tokens,
        val_scene_tokens,
    )