"""Smoke test and visualization for the nuScenes diffusion-policy Dataset."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from diffusion_policy.config import DatasetConfig
from diffusion_policy.dataset import build_train_val_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="v1.0-mini")
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    config = DatasetConfig(dataroot=args.dataroot, version=args.version)
    train_dataset, validation_dataset, train_scenes, validation_scenes = (
        build_train_val_datasets(config)
    )
    assert train_scenes.isdisjoint(validation_scenes)
    assert len(train_dataset) > 0
    assert len(validation_dataset) > 0
    print(f"train scenes: {len(train_scenes)}")
    print(f"validation scenes: {len(validation_scenes)}")

    sample = train_dataset[0]
    print(f"train length: {len(train_dataset)}")
    print(f"validation length: {len(validation_dataset)}")
    print(f"keys: {sorted(sample)}")
    for key, value in sample.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(f"{key} shape: {tuple(shape)}")
    target_actions = sample["target_actions"].numpy()
    assert sample["history_speed"].shape == (config.obs_len,)
    assert sample["history_curvature"].shape == (config.obs_len,)
    assert sample["ego_state"].shape == (7,)
    assert target_actions.shape == (config.fut_len, 2)
    assert np.isfinite(target_actions).all()
    print(f"history_speed:\n{sample['history_speed'].numpy()}")
    print(f"history_curvature:\n{sample['history_curvature'].numpy()}")
    print(f"target_actions:\n{target_actions}")
    print(f"train/validation scenes overlap: {train_scenes & validation_scenes}")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    history_speed = sample["history_speed"].numpy()
    history_curvature = sample["history_curvature"].numpy()
    axes[0].plot(np.arange(config.obs_len), history_speed, label="history speed")
    axes[0].plot(
        np.arange(config.obs_len, config.obs_len + config.fut_len),
        target_actions[:, 0],
        label="future speed",
    )
    axes[0].legend()
    axes[0].set_ylabel("speed")
    axes[1].plot(np.arange(config.obs_len), history_curvature, label="history curvature")
    axes[1].plot(
        np.arange(config.obs_len, config.obs_len + config.fut_len),
        target_actions[:, 1],
        label="future curvature",
    )
    axes[1].legend()
    axes[1].set_xlabel("trajectory step")
    axes[1].set_ylabel("curvature")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()