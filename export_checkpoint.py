"""Package trained diffusion-policy artifacts into a compact ZIP file."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export diffusion-policy checkpoint artifacts.")
    parser.add_argument("--output-dir", default=Path("/kaggle/working/diffusion_outputs"), type=Path)
    parser.add_argument(
        "--zip-path",
        default=None,
        type=Path,
        help="Defaults to <output-dir>/diffusion_policy_checkpoint.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    zip_path = args.zip_path or (output_dir / "diffusion_policy_checkpoint.zip")
    required_files = [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "normalization_stats.json",
        output_dir / "config.json",
        output_dir / "results" / "training_history.json",
        output_dir / "results" / "training_summary.json",
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Cannot export checkpoint; missing files:\n{missing_text}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in required_files:
            archive.write(path, arcname=path.relative_to(output_dir))
    print(f"Exported diffusion checkpoint ZIP: {zip_path}")


if __name__ == "__main__":
    main()
