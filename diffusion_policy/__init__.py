"""Utilities for preparing future diffusion-policy training targets."""

from typing import Any

__all__ = [
	"DiffusionNormalizer",
	"DrivingConditionEncoder",
	"NormalizedDiffusionDataset",
	"NuScenesDiffusionDataset",
	"build_train_val_datasets",
	"collate_diffusion_batch",
	"get_future_speed_curvature",
]


def __getattr__(name: str) -> Any:
	if name == "get_future_speed_curvature":
		from .targets import get_future_speed_curvature

		return get_future_speed_curvature
	if name in {"NuScenesDiffusionDataset", "build_train_val_datasets", "collate_diffusion_batch"}:
		from .dataset import NuScenesDiffusionDataset, build_train_val_datasets, collate_diffusion_batch

		return {
			"NuScenesDiffusionDataset": NuScenesDiffusionDataset,
			"build_train_val_datasets": build_train_val_datasets,
			"collate_diffusion_batch": collate_diffusion_batch,
		}[name]
	if name in {"DiffusionNormalizer", "NormalizedDiffusionDataset"}:
		from .normalization import DiffusionNormalizer, NormalizedDiffusionDataset

		return {
			"DiffusionNormalizer": DiffusionNormalizer,
			"NormalizedDiffusionDataset": NormalizedDiffusionDataset,
		}[name]
	if name == "DrivingConditionEncoder":
		from .conditioning import DrivingConditionEncoder

		return DrivingConditionEncoder
	raise AttributeError(name)