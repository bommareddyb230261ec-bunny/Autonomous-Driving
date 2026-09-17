"""Conditional diffusion-policy network for speed/curvature trajectories."""

import math

import torch
from torch import Tensor, nn

from diffusion_policy.config import (
    ACTION_DIM,
    CONDITION_DIM,
    DIFFUSION_HIDDEN_DIM,
    DIFFUSION_NUM_LAYERS,
    FUT_LEN,
)


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal diffusion timestep embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [B]; got {tuple(timesteps.shape)}")
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        dtype = torch.float32

        if half_dim == 0:
            return timesteps.to(dtype=dtype).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(half_dim, device=device, dtype=dtype)
            * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        angles = timesteps.to(dtype=dtype).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ConditionalResidualBlock1D(nn.Module):
    """Small 1D residual block conditioned on timestep and driving context."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        num_groups = 8 if hidden_dim % 8 == 0 else 1
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        self.activation = nn.GELU()
        self.timestep_projection = nn.Linear(hidden_dim, hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, timestep_embedding: Tensor, condition_embedding: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, C, T]; got {tuple(x.shape)}")
        residual = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.activation(y)
        y = y + self.timestep_projection(timestep_embedding).unsqueeze(-1)
        y = y + self.condition_projection(condition_embedding).unsqueeze(-1)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.activation(y)
        return residual + y


class DrivingDiffusionPolicy(nn.Module):
    """Predict diffusion noise for future speed/curvature action sequences."""

    def __init__(
        self,
        fut_len: int = FUT_LEN,
        action_dim: int = ACTION_DIM,
        condition_dim: int = CONDITION_DIM,
        hidden_dim: int = DIFFUSION_HIDDEN_DIM,
        num_layers: int = DIFFUSION_NUM_LAYERS,
    ) -> None:
        super().__init__()
        if fut_len <= 0:
            raise ValueError("fut_len must be positive")
        if action_dim <= 0 or condition_dim <= 0 or hidden_dim <= 0:
            raise ValueError("action_dim, condition_dim, and hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.fut_len = fut_len
        self.action_dim = action_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.action_embedding = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
        self.timestep_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            ConditionalResidualBlock1D(hidden_dim) for _ in range(num_layers)
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(num_groups=8 if hidden_dim % 8 == 0 else 1, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1),
        )

    def _validate_inputs(self, noisy_actions: Tensor, timesteps: Tensor, condition: Tensor) -> None:
        if noisy_actions.ndim != 3:
            raise ValueError(
                f"noisy_actions must have shape [B, {self.fut_len}, {self.action_dim}]; "
                f"got {tuple(noisy_actions.shape)}"
            )
        expected_action_shape = (noisy_actions.shape[0], self.fut_len, self.action_dim)
        if noisy_actions.shape != expected_action_shape:
            raise ValueError(
                f"noisy_actions must have shape {expected_action_shape}; "
                f"got {tuple(noisy_actions.shape)}"
            )
        if timesteps.shape != (noisy_actions.shape[0],):
            raise ValueError(f"timesteps must have shape [B]; got {tuple(timesteps.shape)}")
        if condition.shape != (noisy_actions.shape[0], self.condition_dim):
            raise ValueError(
                f"condition must have shape [B, {self.condition_dim}]; "
                f"got {tuple(condition.shape)}"
            )
        if noisy_actions.device != timesteps.device or noisy_actions.device != condition.device:
            raise ValueError("noisy_actions, timesteps, and condition must share device")
        if not noisy_actions.is_floating_point() or not condition.is_floating_point():
            raise TypeError("noisy_actions and condition must use floating-point dtypes")

    def forward(self, noisy_actions: Tensor, timesteps: Tensor, condition: Tensor) -> Tensor:
        self._validate_inputs(noisy_actions, timesteps, condition)
        x = noisy_actions.transpose(1, 2)
        x = self.action_embedding(x)
        timestep_embedding = self.timestep_embedding(timesteps)
        condition_embedding = self.condition_embedding(condition)

        for block in self.blocks:
            x = block(x, timestep_embedding, condition_embedding)

        predicted_noise = self.output_projection(x).transpose(1, 2)
        expected_shape = noisy_actions.shape
        if predicted_noise.shape != expected_shape:
            raise RuntimeError(
                f"model returned {tuple(predicted_noise.shape)}, expected {tuple(expected_shape)}"
            )
        return predicted_noise
