"""Utilities for preparing future diffusion-policy training targets."""

from typing import Any

__all__ = [
	"DiffusionNormalizer",
	"DrivingConditionEncoder",
	"NormalizedDiffusionDataset",
	"NuScenesConditioningDataset",
	"NuScenesDiffusionDataset",
	"build_test_conditioning_dataset",
	"build_train_val_datasets",
	"collate_diffusion_batch",
	"get_future_speed_curvature",
	"validate_nuscenes_dataroot",
]


def __getattr__(name: str) -> Any:
	if name == "get_future_speed_curvature":
		from .targets import get_future_speed_curvature

		return get_future_speed_curvature
	if name in {
		"NuScenesConditioningDataset",
		"NuScenesDiffusionDataset",
		"build_test_conditioning_dataset",
		"build_train_val_datasets",
		"collate_diffusion_batch",
		"validate_nuscenes_dataroot",
	}:
		from .dataset import (
			NuScenesConditioningDataset,
			NuScenesDiffusionDataset,
			build_test_conditioning_dataset,
			build_train_val_datasets,
			collate_diffusion_batch,
			validate_nuscenes_dataroot,
		)

		return {
			"NuScenesConditioningDataset": NuScenesConditioningDataset,
			"NuScenesDiffusionDataset": NuScenesDiffusionDataset,
			"build_test_conditioning_dataset": build_test_conditioning_dataset,
			"build_train_val_datasets": build_train_val_datasets,
			"collate_diffusion_batch": collate_diffusion_batch,
			"validate_nuscenes_dataroot": validate_nuscenes_dataroot,
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
