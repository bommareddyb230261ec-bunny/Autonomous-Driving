"""Configuration for the diffusion-policy data preparation stage."""

from dataclasses import dataclass
from pathlib import Path


OBS_LEN = 10
FUT_LEN = 10
DT = 0.5
TRAIN_SPLIT = 0.8
SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 0
EGO_STATE_DIM = 7
CONDITION_DIM = 256
CONDITION_HIDDEN_DIM = 256
ACTION_DIM = 2
DIFFUSION_TIMESTEPS = 1000
DIFFUSION_HIDDEN_DIM = 128
DIFFUSION_NUM_LAYERS = 3


@dataclass(frozen=True)
class DatasetConfig:
    dataroot: Path = Path("v1.0-mini")
    version: str = "v1.0-mini"
    obs_len: int = OBS_LEN
    fut_len: int = FUT_LEN
    dt: float = DT
    train_split: float = TRAIN_SPLIT
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    semantic_cache: Path | None = None
    normalization_stats_path: Path = Path("diffusion_policy/normalization_stats.json")
