"""Plot experiment results from all_results.csv."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import BLUR_KERNELS, JPEG_QUALITIES, NOISE_SIGMAS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_results(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def plot_line(df: pd.DataFrame, prefix: str, levels: list[int], title: str, out_path: Path) -> None:
    rows = []
    for level in levels:
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
    worst_rows = []
    for group, name in [
        ("JPEG", "jpeg_q10"),
        ("Blur", "blur_k11"),
        ("Noise", "noise_s40"),
    ]:
        match = df[df["name"] == name]
        if not match.empty:
            worst_rows.append((group, float(match.iloc[0]["auc"])))

    labels, values = zip(*worst_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#E45756"])
    ax.set_title("Worst-case AUC by degradation type")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center")
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
    plot_line(df, "jpeg_q", JPEG_QUALITIES, "JPEG quality vs AUC", args.output_dir / "jpeg_auc.png")
    plot_line(df, "blur_k", BLUR_KERNELS, "Blur kernel vs AUC", args.output_dir / "blur_auc.png")
    plot_line(df, "noise_s", NOISE_SIGMAS, "Noise sigma vs AUC", args.output_dir / "noise_auc.png")
    plot_worst_bar(df, args.output_dir / "worst_case_auc.png")
    print(f"figures saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
