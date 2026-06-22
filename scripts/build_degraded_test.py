"""Apply degradations to images listed in the test split."""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from adversarial import build_adversarial_sets
from config import RANDOM_SEED
from degrade import apply_degradation, degradation_folders

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_split(split_path: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in split_path.read_text().splitlines():
        if not line.strip():
            continue
        path_str, label = line.rsplit(",", 1)
        rows.append((PROJECT_ROOT / path_str, label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "test.txt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "degraded",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "full_c23.p",
        help="Required for FGSM adversarial sets",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: first N images")
    parser.add_argument("--skip-adversarial", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    samples = read_split(args.split)
    if args.limit is not None:
        samples = samples[: args.limit]

    rng = np.random.default_rng(args.seed)
    folders = degradation_folders(include_adversarial=False)

    for folder_name, ops in folders.items():
        out_dir = args.output_root / folder_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        split_lines = []
        for src_path, label in tqdm(samples, desc=folder_name):
            image = cv2.imread(str(src_path))
            if image is None:
                raise FileNotFoundError(src_path)

            degraded = image
            for kind, level in ops:
                degraded = apply_degradation(degraded, kind, level, rng=rng)

            rel_name = src_path.relative_to(PROJECT_ROOT / "data" / "frames")
            dst_path = out_dir / rel_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_path), degraded)

            split_lines.append(f"data/degraded/{folder_name}/{rel_name},{label}")

        split_path = PROJECT_ROOT / "data" / "splits" / f"test_{folder_name}.txt"
        split_path.write_text("\n".join(split_lines) + "\n")

    if not args.skip_adversarial:
        print("Building FGSM adversarial sets (label-aware evasion)...")
        build_adversarial_sets(
            samples=samples,
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            project_root=PROJECT_ROOT,
        )

    total = len(folders) + (0 if args.skip_adversarial else 4)
    print(f"built {total} degraded test sets under {args.output_root}")


if __name__ == "__main__":
    main()
