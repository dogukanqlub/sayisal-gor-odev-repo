"""Evaluate Xception deepfake detector on a split file."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FF_CLASSIFICATION = PROJECT_ROOT / "_ff_repo" / "classification"
sys.path.insert(0, str(FF_CLASSIFICATION))

from network.models import TransferModel  # noqa: E402

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

LABEL_TO_ID = {"real": 0, "fake": 1}


def load_model(checkpoint_path: Path) -> nn.Module:
    model = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
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


def predict_image(model: nn.Module, image_path: Path) -> tuple[int, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(PILImage.fromarray(image)).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    fake_prob = float(probs[1])
    return pred, fake_prob


def compute_metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc(fpr, tpr),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["real", "fake"])
    ax.set_yticks([0, 1], labels=["real", "fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "full_c23.p",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "test.txt",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "baseline.csv",
    )
    parser.add_argument(
        "--output-cm",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "baseline_cm.png",
    )
    parser.add_argument("--name", type=str, default="baseline")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    samples = read_split(args.split)

    y_true, y_pred, y_score = [], [], []
    for image_path, label in tqdm(samples, desc="eval"):
        pred, fake_prob = predict_image(model, image_path)
        y_true.append(label)
        y_pred.append(pred)
        y_score.append(fake_prob)

    metrics = compute_metrics(y_true, y_pred, y_score)
    save_confusion_matrix(y_true, y_pred, args.output_cm)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "split", "accuracy", "auc", "f1", "precision", "recall"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({"name": args.name, "split": args.split.name, **metrics})

    print(f"{args.name} on {args.split.name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"saved csv -> {args.output_csv}")
    print(f"saved cm  -> {args.output_cm}")


if __name__ == "__main__":
    main()
