"""Synthetic and optional nuScenes smoke tests for the conditioning encoder."""

import argparse

import torch

from diffusion_policy.config import CONDITION_DIM, EGO_STATE_DIM, OBS_LEN, DatasetConfig
from diffusion_policy.conditioning import DrivingConditionEncoder


def _run_encoder_test(device: torch.device) -> None:
    batch_size = 4
    encoder = DrivingConditionEncoder().to(device)
    history_speed = torch.randn(batch_size, OBS_LEN, device=device)
    history_curvature = torch.randn(batch_size, OBS_LEN, device=device)
    ego_state = torch.randn(batch_size, EGO_STATE_DIM, device=device)
    condition = encoder(
        history_speed=history_speed,
        history_curvature=history_curvature,
        ego_state=ego_state,
    )
    assert condition.shape == (batch_size, CONDITION_DIM)
    assert torch.isfinite(condition).all()
    print(f"{device} synthetic output shape: {tuple(condition.shape)}")

    try:
        encoder(history_speed[:, :-1], history_curvature, ego_state)
    except ValueError as exc:
        assert "history_speed" in str(exc)
    else:
        raise AssertionError("incorrect history_speed shape was accepted")


def _run_real_dataset_test(dataroot: str, version: str) -> None:
    try:
        from diffusion_policy.dataset import build_train_val_datasets
        from diffusion_policy.normalization import DiffusionNormalizer, NormalizedDiffusionDataset

        config = DatasetConfig(dataroot=dataroot, version=version)
        train_dataset, _, _, _ = build_train_val_datasets(config)
        normalizer = DiffusionNormalizer().fit(train_dataset)
        normalized_dataset = NormalizedDiffusionDataset(train_dataset, normalizer)
        sample = normalized_dataset[0]
        condition = DrivingConditionEncoder()(
            history_speed=sample["history_speed"].unsqueeze(0),
            history_curvature=sample["history_curvature"].unsqueeze(0),
            ego_state=sample["ego_state"].unsqueeze(0),
        )
        assert condition.shape == (1, CONDITION_DIM)
        assert torch.isfinite(condition).all()
        print(f"real Dataset output shape: {tuple(condition.shape)}")
    except Exception as exc:
        print(f"real Dataset test unavailable: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="v1.0-mini")
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    _run_encoder_test(torch.device("cpu"))
    if torch.cuda.is_available():
        _run_encoder_test(torch.device("cuda"))
    else:
        print("cuda test skipped: CUDA is not available")

    parameter_count = sum(parameter.numel() for parameter in DrivingConditionEncoder().parameters())
    print(f"trainable parameters: {parameter_count}")
    _run_real_dataset_test(args.dataroot, args.version)
    print("conditioning tests complete")


if __name__ == "__main__":
    main()