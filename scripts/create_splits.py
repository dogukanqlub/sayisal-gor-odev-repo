"""Create train/val/test splits by video (no frame leakage)."""

import argparse
import random
from pathlib import Path


def video_ids(frames_dir: Path) -> list[str]:
    ids = set()
    for path in frames_dir.glob("*.jpg"):
        ids.add(path.name.split("_", 1)[0])
    return sorted(ids)


def split_videos(video_list: list[str], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = video_list[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])
    return train_ids, val_ids, test_ids, n_train, n_val, n_test


def collect_rows(frames_dir: Path, label: str, video_ids_set: set[str]) -> list[str]:
    rows = []
    for path in sorted(frames_dir.glob("*.jpg")):
        video_id = path.name.split("_", 1)[0]
        if video_id in video_ids_set:
            rows.append(f"{path},{label}")
    return rows


def write_split(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real_videos = video_ids(args.frames_root / "real")
    fake_videos = video_ids(args.frames_root / "fake")

    real_train, real_val, real_test, *_ = split_videos(
        real_videos, args.train_ratio, args.val_ratio, args.seed
    )
    fake_train, fake_val, fake_test, *_ = split_videos(
        fake_videos, args.train_ratio, args.val_ratio, args.seed + 1
    )

    splits = {
        "train": (real_train | fake_train, "train"),
        "val": (real_val | fake_val, "val"),
        "test": (real_test | fake_test, "test"),
    }

    stats = {}
    for split_name, (video_ids_set, _) in [
        ("train", (real_train | fake_train, None)),
        ("val", (real_val | fake_val, None)),
        ("test", (real_test | fake_test, None)),
    ]:
        real_rows = collect_rows(args.frames_root / "real", "real", real_train if split_name == "train" else real_val if split_name == "val" else real_test)
        fake_rows = collect_rows(args.frames_root / "fake", "fake", fake_train if split_name == "train" else fake_val if split_name == "val" else fake_test)
        rows = real_rows + fake_rows
        write_split(args.output_dir / f"{split_name}.txt", rows)
        stats[split_name] = {
            "real_videos": len(real_train if split_name == "train" else real_val if split_name == "val" else real_test),
            "fake_videos": len(fake_train if split_name == "train" else fake_val if split_name == "val" else fake_test),
            "real_frames": len(real_rows),
            "fake_frames": len(fake_rows),
            "total_frames": len(rows),
        }

    summary_path = args.output_dir / "split_summary.txt"
    lines = ["Phase 1D split summary", ""]
    for split_name, s in stats.items():
        lines.append(
            f"{split_name}: {s['total_frames']} frames "
            f"({s['real_frames']} real, {s['fake_frames']} fake) | "
            f"videos: {s['real_videos']} real, {s['fake_videos']} fake"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
