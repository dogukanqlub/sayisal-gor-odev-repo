"""Plot experiment results from all_results.csv."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    RESIZE_SCALES,
    SATURATION_FACTORS,
    SHARPEN_STRENGTHS,
)
from degrade import degradation_categories

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fmt_level(value: float | int) -> str:
    return str(value).replace(".", "p")


def load_results(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def plot_line(df: pd.DataFrame, prefix: str, levels: list, title: str, out_path: Path) -> None:
    rows = []
    for level in levels:
        if isinstance(level, float):
            name = f"{prefix}{_fmt_level(level)}"
        else:
            name = f"{prefix}{level}"
        match = df[df["name"] == name]
        if not match.empty:
            rows.append((level, float(match.iloc[0]["auc"])))

    if not rows:
        return

    x, y = zip(*rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Level")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_worst_bar(df: pd.DataFrame, out_path: Path) -> None:
    candidates = [
        ("JPEG", "jpeg_q10"),
        ("Blur", "blur_k11"),
        ("Noise", "noise_s40"),
        ("Resize", "resize_s0p25"),
        ("Gamma", "gamma_g0p6"),
        ("FGSM", "fgsm_e16"),
    ]
    worst_rows = []
    for group, name in candidates:
        match = df[df["name"] == name]
        if not match.empty:
            worst_rows.append((group, float(match.iloc[0]["auc"])))

    if not worst_rows:
        return

    labels, values = zip(*worst_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title("Worst-case AUC by degradation category")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_category_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    categories = degradation_categories()
    rows = []
    for cat_name, names in categories.items():
        for name in names:
            match = df[df["name"] == name]
            if not match.empty:
                rows.append({"category": cat_name, "condition": name, "auc": float(match.iloc[0]["auc"])})

    if not rows:
        return

    table = pd.DataFrame(rows)
    pivot = table.pivot(index="condition", columns="category", values="auc")
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.25)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=7)
    ax.set_title("AUC heatmap by degradation category")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "all_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    out = args.output_dir

    plot_line(df, "jpeg_q", JPEG_QUALITIES, "JPEG quality vs AUC", out / "jpeg_auc.png")
    plot_line(df, "blur_k", BLUR_KERNELS, "Blur kernel vs AUC", out / "blur_auc.png")
    plot_line(df, "noise_s", NOISE_SIGMAS, "Noise sigma vs AUC", out / "noise_auc.png")
    plot_line(df, "resize_s", RESIZE_SCALES, "Resize scale vs AUC", out / "postproc_resize_auc.png")
    plot_line(df, "gamma_g", GAMMA_VALUES, "Gamma vs AUC", out / "postproc_gamma_auc.png")
    plot_line(df, "contrast_a", CONTRAST_ALPHAS, "Contrast vs AUC", out / "postproc_contrast_auc.png")
    plot_line(df, "sharpen_s", SHARPEN_STRENGTHS, "Sharpen vs AUC", out / "postproc_sharpen_auc.png")
    plot_line(df, "median_k", MEDIAN_KERNELS, "Median filter vs AUC", out / "postproc_median_auc.png")
    plot_line(df, "saturation_f", SATURATION_FACTORS, "Saturation vs AUC", out / "postproc_saturation_auc.png")
    plot_line(df, "fgsm_e", FGSM_EPSILONS, "FGSM epsilon vs AUC", out / "adversarial_fgsm_auc.png")
    plot_worst_bar(df, out / "worst_case_auc.png")
    plot_category_heatmap(df, out / "category_heatmap.png")
    print(f"figures saved -> {out}")


if __name__ == "__main__":
    main()
