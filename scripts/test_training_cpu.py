"""Synthetic one-step training smoke test for the diffusion-policy stack."""

from __future__ import annotations

import sys

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:
    print(f"PyTorch is unavailable; run this smoke test in Kaggle instead: {type(exc).__name__}: {exc}")
    sys.exit(1)

from diffusion_policy.conditioning import DrivingConditionEncoder
from diffusion_policy.config import ACTION_DIM, CONDITION_DIM, EGO_STATE_DIM, FUT_LEN, OBS_LEN
from diffusion_policy.model import DrivingDiffusionPolicy
from diffusion_policy.scheduler import DrivingDiffusionScheduler


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    condition_encoder = DrivingConditionEncoder().to(device)
    diffusion_policy = DrivingDiffusionPolicy().to(device)
    scheduler = DrivingDiffusionScheduler()
    optimizer = torch.optim.AdamW(
        list(condition_encoder.parameters()) + list(diffusion_policy.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )

    history_speed = torch.randn(batch_size, OBS_LEN, device=device)
    history_curvature = torch.randn(batch_size, OBS_LEN, device=device)
    ego_state = torch.randn(batch_size, EGO_STATE_DIM, device=device)
    clean_actions = torch.randn(batch_size, FUT_LEN, ACTION_DIM, device=device)
    noise = torch.randn_like(clean_actions)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)

    condition = condition_encoder(history_speed, history_curvature, ego_state)
    assert condition.shape == (batch_size, CONDITION_DIM)
    noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
    predicted_noise = diffusion_policy(noisy_actions, timesteps, condition)
    loss = F.mse_loss(predicted_noise, noise)
    assert predicted_noise.shape == clean_actions.shape
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    print(f"synthetic training smoke test passed on {device}; loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
