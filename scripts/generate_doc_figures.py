"""Generate extra figures for documentation (progression grids)."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FRAMES = ROOT / "data" / "frames" / "real"
DEGRADED = ROOT / "data" / "degraded"
OUT = ROOT / "docs" / "assets" / "figures"

SAMPLE = "033_0000.jpg"


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_grid(title: str, labels: list[str], paths: list[Path], out_name: str) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(3 * len(paths), 3.5))
    if len(paths) == 1:
        axes = [axes]
    for ax, label, path in zip(axes, labels, paths):
        ax.imshow(load(path))
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = FRAMES / SAMPLE

    jpeg_quals = [90, 70, 50, 30, 10]
    save_grid(
        "JPEG sıkıştırma ilerlemesi (aynı kare)",
        [f"q={q}" for q in jpeg_quals],
        [DEGRADED / f"jpeg_q{q}" / "real" / SAMPLE for q in jpeg_quals],
        "progression_jpeg.png",
    )

    blur_kernels = [3, 5, 7, 11]
    save_grid(
        "Gaussian blur ilerlemesi (aynı kare)",
        [f"k={k}" for k in blur_kernels],
        [DEGRADED / f"blur_k{k}" / "real" / SAMPLE for k in blur_kernels],
        "progression_blur.png",
    )

    noise_sigmas = [5, 10, 20, 40]
    save_grid(
        "Gaussian gürültü ilerlemesi (aynı kare)",
        [f"σ={s}" for s in noise_sigmas],
        [DEGRADED / f"noise_s{s}" / "real" / SAMPLE for s in noise_sigmas],
        "progression_noise.png",
    )

    real_fake_labels = ["Gerçek (real)", "Sahte (fake)"]
    fake_sample = next((ROOT / "data" / "frames" / "fake").glob("*.jpg"))
    save_grid(
        "Veri seti örnekleri",
        real_fake_labels,
        [base, fake_sample],
        "real_vs_fake.png",
    )

    overview_labels = ["Orijinal", "JPEG q10", "Blur k11", "Noise σ40"]
    overview_paths = [
        base,
        DEGRADED / "jpeg_q10" / "real" / SAMPLE,
        DEGRADED / "blur_k11" / "real" / SAMPLE,
        DEGRADED / "noise_s40" / "real" / SAMPLE,
    ]
    save_grid("En ağır bozulma seviyeleri", overview_labels, overview_paths, "worst_degradations.png")


if __name__ == "__main__":
    main()
