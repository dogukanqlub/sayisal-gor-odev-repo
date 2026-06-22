"""Compare baseline inference vs TTA solution on critical splits."""

import argparse
import csv
import sys
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


def fake_probability(model: nn.Module, image_bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(PILImage.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return float(probs[1])


def predict_baseline(model: nn.Module, image_path: Path) -> tuple[int, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    fake_prob = fake_probability(model, image)
    return int(fake_prob >= 0.5), fake_prob


def predict_tta(model: nn.Module, image_path: Path) -> tuple[int, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    variants = [
        image,
        apply_jpeg(image, 70),
        apply_blur(image, 3),
        apply_denoise_bilateral(image),
    ]
    scores = [fake_probability(model, v) for v in variants]
    fake_prob = float(np.mean(scores))
    return int(fake_prob >= 0.5), fake_prob


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc(fpr, tpr),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_split(model: nn.Module, split_path: Path, method: str) -> dict:
    predict_fn = predict_baseline if method == "baseline" else predict_tta
    samples = read_split(split_path)
    y_true, y_pred, y_score = [], [], []
    for image_path, label in tqdm(samples, desc=f"{method}:{split_path.name}"):
        pred, fake_prob = predict_fn(model, image_path)
        y_true.append(label)
        y_pred.append(pred)
        y_score.append(fake_prob)
    return compute_metrics(y_true, y_pred, y_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "full_c23.p")
    parser.add_argument("--splits", nargs="+", default=CRITICAL_SPLITS)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "solution_comparison.csv",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for split_name in args.splits:
        split_path = PROJECT_ROOT / "data" / "splits" / split_name
        for method in ("baseline", "tta"):
            metrics = evaluate_split(model, split_path, method)
            rows.append({"split": split_name, "method": method, **metrics})
            print(f"{split_name} [{method}] AUC={metrics['auc']:.4f}")

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
