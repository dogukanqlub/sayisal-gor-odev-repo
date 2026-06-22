"""Generate extra figures for documentation (progression grids)."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FRAMES = ROOT / "data" / "frames" / "real"
DEGRADED = ROOT / "data" / "degraded"
OUT = ROOT / "docs" / "assets" / "figures"

SAMPLE = "033_0000.jpg"


def load(path: Path):
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
        "JPEG sıkıştırma ilerlemesi",
        [f"q={q}" for q in jpeg_quals],
        [DEGRADED / f"jpeg_q{q}" / "real" / SAMPLE for q in jpeg_quals],
        "progression_jpeg.png",
    )

    resize_scales = ["0.75", "0.5", "0.25"]
    save_grid(
        "Post-processing: resize (down-up)",
        [f"s={s}" for s in resize_scales],
        [DEGRADED / f"resize_s{s.replace('.', 'p')}" / "real" / SAMPLE for s in resize_scales],
        "progression_resize.png",
    )

    gamma_vals = [0.6, 0.8, 1.2, 1.4]
    save_grid(
        "Post-processing: gamma düzeltme",
        [f"g={g}" for g in gamma_vals],
        [DEGRADED / f"gamma_g{str(g).replace('.', 'p')}" / "real" / SAMPLE for g in gamma_vals],
        "progression_gamma.png",
    )

    fgsm_eps = [4, 8, 16, 32]
    save_grid(
        "Adversarial: FGSM saldırısı",
        [f"ε={e}" for e in fgsm_eps],
        [DEGRADED / f"fgsm_e{e}" / "real" / SAMPLE for e in fgsm_eps],
        "progression_fgsm.png",
    )

    fake_sample = next((ROOT / "data" / "frames" / "fake").glob("*.jpg"))
    save_grid(
        "Veri seti örnekleri",
        ["Gerçek (real)", "Sahte (fake)"],
        [base, fake_sample],
        "real_vs_fake.png",
    )

    overview_labels = ["Orijinal", "Noise σ40", "Resize 0.25", "FGSM ε16"]
    overview_paths = [
        base,
        DEGRADED / "noise_s40" / "real" / SAMPLE,
        DEGRADED / "resize_s0p25" / "real" / SAMPLE,
        DEGRADED / "fgsm_e16" / "real" / SAMPLE,
    ]
    save_grid("En zor koşullar", overview_labels, overview_paths, "worst_degradations.png")


if __name__ == "__main__":
    main()
