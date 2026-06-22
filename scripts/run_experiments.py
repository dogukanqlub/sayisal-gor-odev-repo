"""Evaluate model on baseline and all degraded test splits."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from degrade import degradation_folders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = PROJECT_ROOT / "scripts" / "evaluate.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "full_c23.p",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "all_results.csv",
    )
    args = parser.parse_args()

    if args.output_csv.exists():
        args.output_csv.unlink()

    folders = degradation_folders(include_adversarial=True)
    rows = []

    for folder_name in folders:
        split_path = PROJECT_ROOT / "data" / "splits" / f"test_{folder_name}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        cmd = [
            sys.executable,
            str(EVALUATE),
            "--checkpoint",
            str(args.checkpoint),
            "--split",
            str(split_path),
            "--output-csv",
            str(args.output_csv),
            "--output-cm",
            str(PROJECT_ROOT / "results" / "figures" / f"cm_{folder_name}.png"),
            "--name",
            folder_name,
        ]
        subprocess.run(cmd, check=True)

    print(f"all experiments saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
