"""Ground-truth target extraction for the initial diffusion-policy interface."""

from collections.abc import Sequence

import numpy as np

from utils import EstimateCurvatureFromTrajectory


def get_future_speed_curvature(
    current_ego_position: Sequence[float],
    future_ego_trajectory: Sequence[Sequence[float]],
    dt: float,
    future_len: int,
    min_segment_distance: float = 1e-6,
) -> np.ndarray:
    """Return future ``[speed, curvature]`` targets in world coordinates.

    Each action step is aligned with one future pose. Speed is the distance
    from the previous pose divided by ``dt``. Curvature is estimated from the
    current pose plus all future poses, then aligned to the future poses.

    Invalid or under-sampled windows raise ``ValueError`` so a dataset caller
    can skip them explicitly instead of silently manufacturing labels.
    """
    if future_len <= 0:
        raise ValueError("future_len must be positive")
    if future_len < 2:
        raise ValueError("future_len must be at least 2 for curvature estimation")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive number")
    if not np.isfinite(min_segment_distance) or min_segment_distance < 0:
        raise ValueError("min_segment_distance must be finite and non-negative")

    current = np.asarray(current_ego_position, dtype=float)
    future = np.asarray(future_ego_trajectory, dtype=float)
    if current.ndim != 1 or current.shape[0] < 3:
        raise ValueError("current_ego_position must contain world XYZ coordinates")
    if future.ndim != 2 or future.shape[1] < 3:
        raise ValueError("future_ego_trajectory must have shape (N, >=3)")
    if future.shape[0] < future_len:
        raise ValueError(
            f"future_ego_trajectory needs {future_len} points; got {future.shape[0]}"
        )

    future = future[:future_len, :3]
    current = current[:3]
    trajectory = np.vstack((current, future))
    if not np.isfinite(trajectory).all():
        raise ValueError("ego trajectory contains NaN or Inf values")

    displacements = np.diff(trajectory, axis=0)
    segment_distances = np.linalg.norm(displacements, axis=1)
    if np.any(segment_distances <= min_segment_distance):
        raise ValueError("ego trajectory contains a zero or near-zero movement segment")

    future_speeds = segment_distances / dt
    try:
        estimated_curvatures = EstimateCurvatureFromTrajectory(trajectory)
    except (IndexError, ValueError, FloatingPointError) as exc:
        raise ValueError("curvature estimation failed for ego trajectory") from exc
    future_curvatures = estimated_curvatures[1:]
    future_actions = np.column_stack((future_speeds, future_curvatures))
    if future_actions.shape != (future_len, 2):
        raise ValueError(f"unexpected target shape: {future_actions.shape}")
    if not np.isfinite(future_actions).all():
        raise ValueError("future targets contain NaN or Inf values")
    return future_actions