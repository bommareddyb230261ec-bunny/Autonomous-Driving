"""Verify training-only normalization and numerical round trips."""

import argparse

import numpy as np

from diffusion_policy.config import DatasetConfig
from diffusion_policy.dataset import build_train_val_datasets
from diffusion_policy.normalization import (
    DiffusionNormalizer,
    NormalizedDiffusionDataset,
    summarize_dataset,
)


def _print_statistics(title: str, statistics: dict) -> None:
    print(title)
    for field, values in statistics.items():
        print(f"{field}:")
        print(f"  mean = {values['mean']}")
        print(f"  std = {values['std']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="v1.0-mini")
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    config = DatasetConfig(dataroot=args.dataroot, version=args.version)
    train_dataset, validation_dataset, _, _ = build_train_val_datasets(config)
    normalizer = DiffusionNormalizer().fit(train_dataset)
    normalizer.save(
        config.normalization_stats_path,
        obs_len=config.obs_len,
        fut_len=config.fut_len,
    )
    normalized_train = NormalizedDiffusionDataset(train_dataset, normalizer)

    train_statistics = normalizer.statistics()
    validation_statistics = summarize_dataset(validation_dataset)
    _print_statistics("Training statistics used for normalization:", train_statistics)
    _print_statistics("Validation statistics (reported only, never fitted):", validation_statistics)

    max_error = 0.0
    for index in range(min(3, len(train_dataset))):
        raw_sample = train_dataset[index]
        normalized_sample = normalized_train[index]
        assert np.isfinite(normalized_sample["history_speed"].numpy()).all()
        assert np.isfinite(normalized_sample["history_curvature"].numpy()).all()
        assert np.isfinite(normalized_sample["ego_state"].numpy()).all()
        assert np.isfinite(normalized_sample["target_actions"].numpy()).all()
        reconstructed = normalizer.inverse_transform(normalized_sample)
        for field in ("history_speed", "history_curvature", "ego_state", "target_actions"):
            error = np.max(
                np.abs(reconstructed[field].numpy() - raw_sample[field].numpy())
            )
            max_error = max(max_error, float(error))
    assert max_error < 1e-5, f"round-trip error too large: {max_error}"

    print("Normalization complete")
    print(f"Statistics saved to: {config.normalization_stats_path}")
    print(f"Round-trip reconstruction error: {max_error:.8f}")


if __name__ == "__main__":
    main()