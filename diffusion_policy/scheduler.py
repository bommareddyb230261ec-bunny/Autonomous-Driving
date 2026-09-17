"""Diffusion scheduler wrapper for driving action trajectories."""

from torch import Tensor

from diffusion_policy.config import DIFFUSION_TIMESTEPS


class DrivingDiffusionScheduler:
    """Thin wrapper around diffusers.DDPMScheduler for training-time noise."""

    def __init__(
        self,
        num_train_timesteps: int = DIFFUSION_TIMESTEPS,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        try:
            from diffusers import DDPMScheduler
        except Exception as exc:  # pragma: no cover - exercised only when env is broken.
            raise ImportError(f"failed to import diffusers.DDPMScheduler: {exc}") from exc

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type

    def add_noise(self, actions: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        if actions.shape != noise.shape:
            raise ValueError(f"actions and noise must share shape; got {actions.shape} and {noise.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B, T, A]; got {tuple(actions.shape)}")
        if timesteps.shape != (actions.shape[0],):
            raise ValueError(f"timesteps must have shape [B]; got {tuple(timesteps.shape)}")
        if actions.device != noise.device or actions.device != timesteps.device:
            raise ValueError("actions, noise, and timesteps must share device")
        return self.scheduler.add_noise(actions, noise, timesteps)
