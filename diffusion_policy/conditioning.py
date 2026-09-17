"""Numerical driving-context encoder for future diffusion-policy conditioning."""

import torch
from torch import Tensor, nn

from diffusion_policy.config import (
    CONDITION_DIM,
    CONDITION_HIDDEN_DIM,
    EGO_STATE_DIM,
    OBS_LEN,
)


class DrivingConditionEncoder(nn.Module):
    """Encode normalized history and ego state into one conditioning vector."""

    def __init__(
        self,
        obs_len: int = OBS_LEN,
        ego_state_dim: int = EGO_STATE_DIM,
        hidden_dim: int = CONDITION_HIDDEN_DIM,
        condition_dim: int = CONDITION_DIM,
    ) -> None:
        super().__init__()
        if obs_len <= 0 or ego_state_dim <= 0:
            raise ValueError("obs_len and ego_state_dim must be positive")
        if hidden_dim <= 0 or condition_dim <= 0:
            raise ValueError("hidden_dim and condition_dim must be positive")
        self.obs_len = obs_len
        self.ego_state_dim = ego_state_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.input_dim = obs_len + obs_len + ego_state_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def _validate_input(self, name: str, value: Tensor, feature_dim: int) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [B, {feature_dim}]; got {tuple(value.shape)}")
        if value.shape[1] != feature_dim:
            raise ValueError(f"{name} must have shape [B, {feature_dim}]; got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")

    def forward(
        self,
        history_speed: Tensor,
        history_curvature: Tensor,
        ego_state: Tensor,
    ) -> Tensor:
        self._validate_input("history_speed", history_speed, self.obs_len)
        self._validate_input("history_curvature", history_curvature, self.obs_len)
        self._validate_input("ego_state", ego_state, self.ego_state_dim)
        batch_size = history_speed.shape[0]
        if history_curvature.shape[0] != batch_size or ego_state.shape[0] != batch_size:
            raise ValueError("history_speed, history_curvature, and ego_state must share batch size")
        if history_speed.device != history_curvature.device or history_speed.device != ego_state.device:
            raise ValueError("history_speed, history_curvature, and ego_state must share device")
        features = torch.cat((history_speed, history_curvature, ego_state), dim=1)
        condition = self.network(features)
        expected_shape = (batch_size, self.condition_dim)
        if condition.shape != expected_shape:
            raise RuntimeError(f"conditioning network returned {tuple(condition.shape)}, expected {expected_shape}")
        return condition