"""Load one nuScenes window and inspect future speed/curvature targets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes

from diffusion_policy.targets import get_future_speed_curvature
from utils import EstimateCurvatureFromTrajectory


OBS_LEN = 10
FUT_LEN = 10
DT = 0.5


def _load_first_valid_window(
    dataroot: Path, version: str
) -> tuple[np.ndarray, np.ndarray]:
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    for scene in nusc.scene:
        poses = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
            poses.append(pose["translation"][:3])
            sample_token = sample["next"]

        positions = np.asarray(poses, dtype=float)
        for current_index in range(OBS_LEN - 1, len(positions) - FUT_LEN):
            current = positions[current_index]
            future = positions[current_index + 1 : current_index + FUT_LEN + 1]
            try:
                get_future_speed_curvature(current, future, DT, FUT_LEN)
            except ValueError:
                continue
            return current, future
    raise RuntimeError("No valid nuScenes window with sufficient movement was found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, default=Path("v1.0-mini"))
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    current, future = _load_first_valid_window(args.dataroot, args.version)
    actions = get_future_speed_curvature(current, future, DT, FUT_LEN)

    assert actions.shape == (FUT_LEN, 2)
    assert np.isfinite(actions).all()
    assert np.all(actions[:, 0] >= 0.0)
    trajectory = np.vstack((current, future))
    expected_speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) / DT
    expected_curvatures = EstimateCurvatureFromTrajectory(trajectory)[1:]
    assert np.allclose(actions[:, 0], expected_speeds)
    assert np.allclose(actions[:, 1], expected_curvatures)
    print("future_actions:")
    print(actions)
    print(f"shape: {actions.shape}")
    print(f"speed min/max: {actions[:, 0].min():.6f} / {actions[:, 0].max():.6f}")
    print(
        "curvature min/max: "
        f"{actions[:, 1].min():.6f} / {actions[:, 1].max():.6f}"
    )
    print(f"finite values: {np.isfinite(actions).all()}")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], "o-")
    axes[0].set_title("Ground-truth future XY")
    axes[0].set_xlabel("world x")
    axes[0].set_ylabel("world y")
    axes[1].plot(np.arange(1, FUT_LEN + 1), actions[:, 0], "o-")
    axes[1].set_title("Future speed")
    axes[1].set_xlabel("future step")
    axes[2].plot(np.arange(1, FUT_LEN + 1), actions[:, 1], "o-")
    axes[2].set_title("Future curvature")
    axes[2].set_xlabel("future step")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()