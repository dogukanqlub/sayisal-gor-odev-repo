"""Compare baseline vs lightweight solution proposals on critical splits."""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from degrade import apply_blur, apply_denoise_bilateral, apply_jpeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FF_CLASSIFICATION = PROJECT_ROOT / "_ff_repo" / "classification"
sys.path.insert(0, str(FF_CLASSIFICATION))

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
LABEL_TO_ID = {"real": 0, "fake": 1}

CRITICAL_SPLITS = [
    "test_baseline.txt",
    "test_noise_s40.txt",
    "test_fgsm_e16.txt",
    "test_blur_k11.txt",
    "test_resize_s0p25.txt",
]

METHODS = ("baseline", "tta", "temporal", "freq_fusion")


def load_model(checkpoint_path: Path) -> nn.Module:
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def read_split(split_path: Path) -> list[tuple[Path, int]]:
    rows = []
    for line in split_path.read_text().splitlines():
        if not line.strip():
            continue
        path_str, label = line.rsplit(",", 1)
        rows.append((PROJECT_ROOT / path_str, LABEL_TO_ID[label]))
    return rows


def video_id(image_path: Path) -> str:
    return image_path.name.split("_", 1)[0]


def fake_probability(model: nn.Module, image_bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(PILImage.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return float(probs[1])


def high_freq_ratio(image_bgr: np.ndarray) -> float:
    """High-frequency energy ratio via FFT magnitude (center mask)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    mask = np.ones_like(magnitude, dtype=np.float32)
    mask[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8] = 0.0
    high = float((magnitude * mask).sum())
    total = float(magnitude.sum()) + 1e-8
    return high / total


def calibrate_freq_weight(
    model: nn.Module,
    val_split: Path,
    hf_min: float,
    hf_max: float,
) -> float:
    samples = read_split(val_split)
    best_w, best_auc = 0.85, -1.0
    cnn_scores, hf_scores, labels = [], [], []

    for image_path, label in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        cnn_scores.append(fake_probability(model, image))
        hf_raw = high_freq_ratio(image)
        hf_norm = (hf_raw - hf_min) / (hf_max - hf_min + 1e-8)
        hf_scores.append(float(np.clip(hf_norm, 0, 1)))
        labels.append(label)

    for w in np.arange(0.55, 0.96, 0.05):
        fused = [w * c + (1 - w) * h for c, h in zip(cnn_scores, hf_scores)]
        fpr, tpr, _ = roc_curve(labels, fused)
        score = auc(fpr, tpr)
        if score > best_auc:
            best_auc = score
            best_w = float(w)
    return best_w


def hf_calibration_bounds(model: nn.Module, val_split: Path) -> tuple[float, float]:
    samples = read_split(val_split)
    values = []
    for image_path, _ in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        values.append(high_freq_ratio(image))
    return float(min(values)), float(max(values))


def frame_scores(
    model: nn.Module,
    samples: list[tuple[Path, int]],
    method: str,
    freq_weight: float,
    hf_min: float,
    hf_max: float,
) -> tuple[list[int], list[int], list[float]]:
    y_true, y_pred, y_score = [], [], []
    for image_path, label in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        if method == "baseline":
            fake_prob = fake_probability(model, image)
        elif method == "tta":
            variants = [image, apply_jpeg(image, 70), apply_blur(image, 3), apply_denoise_bilateral(image)]
            fake_prob = float(np.mean([fake_probability(model, v) for v in variants]))
        elif method == "freq_fusion":
            cnn = fake_probability(model, image)
            hf_norm = (high_freq_ratio(image) - hf_min) / (hf_max - hf_min + 1e-8)
            hf_norm = float(np.clip(hf_norm, 0, 1))
            fake_prob = freq_weight * cnn + (1 - freq_weight) * hf_norm
        else:
            raise ValueError(method)

        y_true.append(label)
        y_score.append(fake_prob)
        y_pred.append(int(fake_prob >= 0.5))
    return y_true, y_pred, y_score


def video_scores(
    model: nn.Module,
    samples: list[tuple[Path, int]],
) -> tuple[list[int], list[int], list[float]]:
    """Temporal ensemble: ortalama fake olasılığı video başına."""
    buckets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for image_path, label in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        buckets[video_id(image_path)].append((label, fake_probability(model, image)))

    y_true, y_pred, y_score = [], [], []
    for vid in sorted(buckets):
        entries = buckets[vid]
        label = entries[0][0]
        fake_prob = float(np.mean([score for _, score in entries]))
        y_true.append(label)
        y_score.append(fake_prob)
        y_pred.append(int(fake_prob >= 0.5))
    return y_true, y_pred, y_score


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    if len(set(y_true)) < 2:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": float("nan"),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc(fpr, tpr),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_split(
    model: nn.Module,
    split_path: Path,
    method: str,
    freq_weight: float,
    hf_min: float,
    hf_max: float,
) -> dict:
    samples = read_split(split_path)
    if method == "temporal":
        y_true, y_pred, y_score = video_scores(model, samples)
    else:
        y_true, y_pred, y_score = frame_scores(
            model, samples, method, freq_weight, hf_min, hf_max
        )
    return compute_metrics(y_true, y_pred, y_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "full_c23.p")
    parser.add_argument("--val-split", type=Path, default=PROJECT_ROOT / "data" / "splits" / "val.txt")
    parser.add_argument("--splits", nargs="+", default=CRITICAL_SPLITS)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "solution_comparison.csv",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    hf_min, hf_max = hf_calibration_bounds(model, args.val_split)
    freq_weight = calibrate_freq_weight(model, args.val_split, hf_min, hf_max)
    print(f"freq_fusion weight (val AUC): w={freq_weight:.2f}, hf_range=[{hf_min:.4f}, {hf_max:.4f}]")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for split_name in args.splits:
        split_path = PROJECT_ROOT / "data" / "splits" / split_name
        for method in METHODS:
            metrics = evaluate_split(model, split_path, method, freq_weight, hf_min, hf_max)
            rows.append({"split": split_name, "method": method, **metrics})
            auc_str = "nan" if np.isnan(metrics["auc"]) else f"{metrics['auc']:.4f}"
            print(f"{split_name} [{method}] AUC={auc_str} recall={metrics['recall']:.4f}")

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "method", "accuracy", "auc", "f1", "precision", "recall"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
