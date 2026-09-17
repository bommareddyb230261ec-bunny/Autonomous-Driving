"""Synthetic forward/noise tests for the conditional diffusion policy."""

from __future__ import annotations

import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - used for local environment diagnosis.
    print(f"failed to import torch: {type(exc).__name__}: {exc}")
    sys.exit(1)

try:
    import diffusers
except Exception as exc:  # pragma: no cover - used for local environment diagnosis.
    print(f"failed to import diffusers: {type(exc).__name__}: {exc}")
    sys.exit(1)

from diffusion_policy.conditioning import DrivingConditionEncoder
from diffusion_policy.config import ACTION_DIM, CONDITION_DIM, FUT_LEN
from diffusion_policy.model import DrivingDiffusionPolicy
from diffusion_policy.scheduler import DrivingDiffusionScheduler


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def _run_forward_test(device: torch.device, batch_size: int) -> None:
    model = DrivingDiffusionPolicy().to(device)
    scheduler = DrivingDiffusionScheduler()
    clean_actions = torch.randn(batch_size, FUT_LEN, ACTION_DIM, device=device)
    condition = torch.randn(batch_size, CONDITION_DIM, device=device)
    noise = torch.randn_like(clean_actions)
    timesteps = torch.randint(
        0,
        scheduler.num_train_timesteps,
        (batch_size,),
        device=device,
    )

    noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
    predicted_noise = model(noisy_actions, timesteps, condition)

    assert predicted_noise.shape == clean_actions.shape
    assert torch.isfinite(predicted_noise).all()
    print(
        f"{device} B={batch_size}: noisy_actions={tuple(noisy_actions.shape)}, "
        f"predicted_noise={tuple(predicted_noise.shape)}"
    )


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"diffusers: {diffusers.__version__}")

    scheduler = DrivingDiffusionScheduler()
    print(f"scheduler timesteps: {scheduler.num_train_timesteps}")
    print(f"scheduler beta_schedule: {scheduler.beta_schedule}")
    print(f"scheduler prediction_type: {scheduler.prediction_type}")

    condition_encoder = DrivingConditionEncoder()
    diffusion_policy = DrivingDiffusionPolicy()
    encoder_total, encoder_trainable = _count_parameters(condition_encoder)
    policy_total, policy_trainable = _count_parameters(diffusion_policy)
    print(f"DrivingConditionEncoder total parameters: {encoder_total}")
    print(f"DrivingConditionEncoder trainable parameters: {encoder_trainable}")
    print(f"DrivingDiffusionPolicy total parameters: {policy_total}")
    print(f"DrivingDiffusionPolicy trainable parameters: {policy_trainable}")
    print(f"combined total parameters: {encoder_total + policy_total}")
    print(f"combined trainable parameters: {encoder_trainable + policy_trainable}")

    for batch_size in (1, 2, 4):
        _run_forward_test(torch.device("cpu"), batch_size)

    if torch.cuda.is_available():
        for batch_size in (1, 2, 4):
            _run_forward_test(torch.device("cuda"), batch_size)
    else:
        print("cuda test skipped: CUDA is not available")

    print("diffusion model tests complete")


if __name__ == "__main__":
    main()
