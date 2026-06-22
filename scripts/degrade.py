"""Image degradation utilities for the robustness experiment."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    BLUR_KERNELS,
    CONTRAST_ALPHAS,
    FGSM_EPSILONS,
    GAMMA_VALUES,
    JPEG_QUALITIES,
    MEDIAN_KERNELS,
    NOISE_SIGMAS,
    RANDOM_SEED,
    RESIZE_SCALES,
    SATURATION_FACTORS,
    SHARPEN_STRENGTHS,
)


def apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        raise RuntimeError(f"JPEG encode failed for quality={quality}")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_noise(image: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    gen = rng or np.random.default_rng()
    noise = gen.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_resize(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    adjusted = image.astype(np.float32) * alpha + 128.0 * (1.0 - alpha)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_sharpen(image: np.ndarray, strength: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_median(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_denoise_bilateral(image: np.ndarray) -> np.ndarray:
    """Light bilateral filter — proposed pre-processing step."""
    return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)


def apply_degradation(
    image: np.ndarray,
    kind: str,
    level: int | float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if kind == "none":
        return image.copy()
    if kind == "jpeg":
        return apply_jpeg(image, int(level))
    if kind == "blur":
        return apply_blur(image, int(level))
    if kind == "noise":
        return apply_noise(image, float(level), rng=rng)
    if kind == "resize":
        return apply_resize(image, float(level))
    if kind == "gamma":
        return apply_gamma(image, float(level))
    if kind == "contrast":
        return apply_contrast(image, float(level))
    if kind == "sharpen":
        return apply_sharpen(image, float(level))
    if kind == "median":
        return apply_median(image, int(level))
    if kind == "saturation":
        return apply_saturation(image, float(level))
    raise ValueError(f"Unknown degradation: {kind}")


def _folder_specs() -> dict[str, list[tuple[str, int | float]]]:
    folders: dict[str, list[tuple[str, int | float]]] = {
        "baseline": [("none", 0)],
    }
    folders.update({f"jpeg_q{q}": [("jpeg", q)] for q in JPEG_QUALITIES})
    folders.update({f"blur_k{k}": [("blur", k)] for k in BLUR_KERNELS})
    folders.update({f"noise_s{s}": [("noise", s)] for s in NOISE_SIGMAS})
    folders.update({f"resize_s{str(s).replace('.', 'p')}": [("resize", s)] for s in RESIZE_SCALES})
    folders.update({f"gamma_g{str(g).replace('.', 'p')}": [("gamma", g)] for g in GAMMA_VALUES})
    folders.update({f"contrast_a{str(a).replace('.', 'p')}": [("contrast", a)] for a in CONTRAST_ALPHAS})
    folders.update({f"sharpen_s{str(s).replace('.', 'p')}": [("sharpen", s)] for s in SHARPEN_STRENGTHS})
    folders.update({f"median_k{k}": [("median", k)] for k in MEDIAN_KERNELS})
    folders.update({f"saturation_f{str(f).replace('.', 'p')}": [("saturation", f)] for f in SATURATION_FACTORS})
    return folders


def degradation_folders(include_adversarial: bool = False) -> dict[str, list[tuple[str, int | float]]]:
    folders = _folder_specs()
    if include_adversarial:
        folders.update({f"fgsm_e{e}": [("fgsm", e)] for e in FGSM_EPSILONS})
    return folders


def degradation_categories() -> dict[str, list[str]]:
    folders = degradation_folders(include_adversarial=True)
    categories: dict[str, list[str]] = {
        "compression": ["baseline"]
        + [f"jpeg_q{q}" for q in JPEG_QUALITIES]
        + [f"blur_k{k}" for k in BLUR_KERNELS]
        + [f"noise_s{s}" for s in NOISE_SIGMAS],
        "postproc": [
            name
            for name in folders
            if name.startswith(("resize_", "gamma_", "contrast_", "sharpen_", "median_", "saturation_"))
        ],
        "adversarial": [f"fgsm_e{e}" for e in FGSM_EPSILONS],
    }
    return categories


def save_samples(
    image_path: Path,
    output_path: Path,
    seed: int = RANDOM_SEED,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    rng = np.random.default_rng(seed)
    samples = [
        ("original", image),
        ("jpeg_q10", apply_jpeg(image, 10)),
        ("blur_k11", apply_blur(image, 11)),
        ("noise_s40", apply_noise(image, 40, rng=rng)),
        ("resize_s0p25", apply_resize(image, 0.25)),
        ("gamma_g0p6", apply_gamma(image, 0.6)),
        ("sharpen_s2p0", apply_sharpen(image, 2.0)),
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
