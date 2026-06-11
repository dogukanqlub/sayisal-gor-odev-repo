"""Extract evenly spaced frames from downloaded FF++ c23 videos."""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_DIRS = {
    "real": "original_sequences/youtube/c23/videos",
    "fake": "manipulated_sequences/Deepfakes/c23/videos",
}


def extract_frames(video_path: Path, output_dir: Path, num_frames: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    indices = [
        int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)
    ]
    saved = 0
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_file = output_dir / f"{video_path.stem}_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_file), frame)
        saved += 1
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--num-frames", type=int, default=10)
    args = parser.parse_args()

    for label, rel_dir in VIDEO_DIRS.items():
        video_dir = args.data_root / rel_dir
        videos = sorted(video_dir.glob("*.mp4"))
        print(f"{label}: {len(videos)} videos")
        for video in tqdm(videos, desc=label):
            out_dir = args.output_root / label
            extract_frames(video, out_dir, args.num_frames)


if __name__ == "__main__":
    main()
