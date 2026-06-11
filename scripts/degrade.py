"""Image degradation utilities for the robustness experiment."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import BLUR_KERNELS, JPEG_QUALITIES, NOISE_SIGMAS


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        raise RuntimeError(f"JPEG encode failed for quality={quality}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def apply_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def degradation_folders() -> dict[str, list[tuple[str, int | float]]]:
    folders: dict[str, list[tuple[str, int | float]]] = {
        "baseline": [("none", 0)],
    }
    folders.update({f"jpeg_q{q}": [("jpeg", q)] for q in JPEG_QUALITIES})
    folders.update({f"blur_k{k}": [("blur", k)] for k in BLUR_KERNELS})
    folders.update({f"noise_s{s}": [("noise", s)] for s in NOISE_SIGMAS})
    return folders


def apply_degradation(image: np.ndarray, kind: str, level: int | float) -> np.ndarray:
    if kind == "none":
        return image.copy()
    if kind == "jpeg":
        return apply_jpeg(image, int(level))
    if kind == "blur":
        return apply_blur(image, int(level))
    if kind == "noise":
        return apply_noise(image, float(level))
    raise ValueError(f"Unknown degradation: {kind}")


def save_samples(
    image_path: Path,
    output_path: Path,
    seed: int = 42,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    np.random.seed(seed)
    samples = [
        ("original", image),
        ("jpeg_q10", apply_jpeg(image, 10)),
        ("blur_k11", apply_blur(image, 11)),
        ("noise_s40", apply_noise(image, 40)),
    ]

    tiles = [sample[1] for sample in samples]
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.zeros((tile_h, tile_w * len(tiles), 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        canvas[:, idx * tile_w : (idx + 1) * tile_w] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

    labels_path = output_path.with_suffix(".txt")
    labels_path.write_text("\n".join(name for name, _ in samples) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/degradation_samples.png"),
    )
    args = parser.parse_args()
    save_samples(args.image, args.output)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
