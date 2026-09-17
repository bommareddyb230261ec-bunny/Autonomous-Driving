"""Training-only normalization for diffusion-policy numerical fields."""

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


NUMERICAL_FIELDS = (
    "history_speed",
    "history_curvature",
    "ego_state",
    "target_actions",
)
_JSON_VERSION = 1


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _summary_reduction(field: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if field in {"history_speed", "history_curvature", "ego_state"}:
        return values.mean(axis=0), values.std(axis=0)
    return values.mean(axis=(0, 1)), values.std(axis=(0, 1))


def summarize_dataset(dataset: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Calculate raw summaries for reporting; this does not fit a normalizer."""
    values = {field: [] for field in NUMERICAL_FIELDS}
    for sample in dataset:
        for field in NUMERICAL_FIELDS:
            values[field].append(_as_numpy(sample[field]))
    if not values["target_actions"]:
        raise ValueError("cannot summarize an empty dataset")
    summaries = {}
    for field, field_values in values.items():
        stacked = np.stack(field_values)
        mean, std = _summary_reduction(field, stacked)
        summaries[field] = {"mean": mean, "std": std}
    return summaries


class DiffusionNormalizer:
    """Standardize numerical Dataset fields using training data only."""

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, np.ndarray]] = {}
        self._safe_std: dict[str, np.ndarray] = {}
        self._protected_dimensions: dict[str, list[int]] = {}

    @property
    def fitted(self) -> bool:
        return len(self._stats) == len(NUMERICAL_FIELDS)

    def fit(self, dataset: Iterable[Mapping[str, Any]]) -> "DiffusionNormalizer":
        summaries = summarize_dataset(dataset)
        self._stats = {}
        self._safe_std = {}
        self._protected_dimensions = {}
        for field in NUMERICAL_FIELDS:
            mean = summaries[field]["mean"]
            std = summaries[field]["std"]
            protected = np.flatnonzero(std < 1e-8).tolist()
            safe_std = np.maximum(std, 1e-8)
            if protected:
                message = (
                    f"Normalization warning: {field} dimensions {protected} "
                    "have std < 1e-8; using 1e-8."
                )
                print(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._stats[field] = {"mean": mean, "std": std}
            self._safe_std[field] = safe_std
            self._protected_dimensions[field] = protected
        return self

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("DiffusionNormalizer.fit() must be called first")

    def _transform_field(self, field: str, value: Any, inverse: bool) -> torch.Tensor:
        self._require_fitted()
        if field not in NUMERICAL_FIELDS:
            raise KeyError(f"unknown numerical field: {field}")
        array = _as_numpy(value)
        stats = self._stats[field]
        if inverse:
            transformed = array * self._safe_std[field] + stats["mean"]
        else:
            transformed = (array - stats["mean"]) / self._safe_std[field]
        if not np.isfinite(transformed).all():
            raise ValueError(f"{field} normalization produced NaN or Inf")
        return torch.as_tensor(transformed, dtype=torch.float32)

    def transform(
        self, sample_or_values: Mapping[str, Any] | Any, field: str | None = None
    ) -> dict[str, Any] | torch.Tensor:
        """Normalize a sample mapping or one named numerical field."""
        if isinstance(sample_or_values, Mapping):
            sample = dict(sample_or_values)
            for numerical_field in NUMERICAL_FIELDS:
                sample[numerical_field] = self._transform_field(
                    numerical_field, sample[numerical_field], inverse=False
                )
            return sample
        if field is None:
            raise ValueError("field is required when transforming individual values")
        return self._transform_field(field, sample_or_values, inverse=False)

    def inverse_transform(
        self, sample_or_values: Mapping[str, Any] | Any, field: str | None = None
    ) -> dict[str, Any] | torch.Tensor:
        """Restore a normalized sample mapping or one named numerical field."""
        if isinstance(sample_or_values, Mapping):
            sample = dict(sample_or_values)
            for numerical_field in NUMERICAL_FIELDS:
                sample[numerical_field] = self._transform_field(
                    numerical_field, sample[numerical_field], inverse=True
                )
            return sample
        if field is None:
            raise ValueError("field is required when inverse-transforming individual values")
        return self._transform_field(field, sample_or_values, inverse=True)

    def statistics(self) -> dict[str, dict[str, Any]]:
        self._require_fitted()
        return {
            field: {
                "mean": stats["mean"].tolist(),
                "std": stats["std"].tolist(),
                "safe_std": self._safe_std[field].tolist(),
                "protected_dimensions": self._protected_dimensions[field],
            }
            for field, stats in self._stats.items()
        }

    def save(
        self,
        path: str | Path,
        obs_len: int,
        fut_len: int,
        feature_names: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_fitted()
        payload = {
            "version": _JSON_VERSION,
            "obs_len": obs_len,
            "fut_len": fut_len,
            "feature_names": feature_names
            or {
                "history_speed": "speed per observation step",
                "history_curvature": "curvature per observation step",
                "ego_state": ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z", "curvature"],
                "target_actions": ["future_speed", "future_curvature"],
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "statistics": self.statistics(),
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class NormalizedDiffusionDataset(Dataset):
    """Explicit normalized view over a raw diffusion-policy Dataset."""

    def __init__(self, dataset: Dataset, normalizer: DiffusionNormalizer):
        if not normalizer.fitted:
            raise ValueError("normalizer must be fitted before wrapping a Dataset")
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.normalizer.transform(self.dataset[index])