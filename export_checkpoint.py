"""Package trained diffusion-policy artifacts into a compact ZIP file."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def export_checkpoint(checkpoint_path: Path, output_path: Path, output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = checkpoint_path.parent.parent
    required_files = [
        checkpoint_path,
        output_dir / "normalization_stats.json",
        output_dir / "config.json",
        output_dir / "training_history.json",
        output_dir / "loss_curve.png",
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Cannot export checkpoint; missing files:\n{missing_text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(checkpoint_path, arcname="best.pt")
        for path in required_files[1:]:
            archive.write(path, arcname=path.name)
    print(f"Exported diffusion checkpoint ZIP: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export diffusion-policy checkpoint artifacts.")
    parser.add_argument("--output-dir", default=Path("/kaggle/working/diffusion_outputs"), type=Path)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="Checkpoint to export. Defaults to <output-dir>/checkpoints/best.pt",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="ZIP path. Defaults to <output-dir>/diffusion_policy_checkpoint.zip",
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        type=Path,
        help="Backward-compatible alias for --output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint or (output_dir / "checkpoints" / "best.pt")
    zip_path = args.output or args.zip_path or (output_dir / "diffusion_policy_checkpoint.zip")
    export_checkpoint(checkpoint_path=checkpoint_path, output_path=zip_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
