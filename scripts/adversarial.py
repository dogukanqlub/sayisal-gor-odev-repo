"""Label-aware FGSM attacks against the Xception deepfake detector."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import FGSM_EPSILONS, RANDOM_SEED

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


def load_model(checkpoint_path: Path) -> nn.Module:
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return TRANSFORM(PILImage.fromarray(rgb)).unsqueeze(0)


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().squeeze(0).cpu()
    x = x * 0.5 + 0.5
    x = torch.clamp(x, 0, 1)
    rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fgsm_attack(
    model: nn.Module,
    image_bgr: np.ndarray,
    label_id: int,
    epsilon: float,
) -> np.ndarray:
    """Non-targeted FGSM: maximize CE loss w.r.t. true label (evasion)."""
    x = bgr_to_tensor(image_bgr)
    x.requires_grad_(True)

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([label_id]))
    loss.backward()

    grad_sign = x.grad.detach().sign()
    # epsilon is in normalized input space; map from pixel budget via /255*2
    eps_norm = (epsilon / 255.0) * 2.0
    x_adv = torch.clamp(x + eps_norm * grad_sign, -1.0, 1.0)
    return tensor_to_bgr(x_adv)


def build_adversarial_sets(
    samples: list[tuple[Path, str]],
    output_root: Path,
    checkpoint: Path,
    project_root: Path,
    epsilons: list[int] | None = None,
) -> None:
    epsilons = epsilons or FGSM_EPSILONS
    model = load_model(checkpoint)
    splits_root = project_root / "data" / "splits"

    for epsilon in epsilons:
        folder_name = f"fgsm_e{epsilon}"
        out_dir = output_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        split_lines: list[str] = []

        for src_path, label in samples:
            image = cv2.imread(str(src_path))
            if image is None:
                raise FileNotFoundError(src_path)

            attacked = fgsm_attack(model, image, LABEL_TO_ID[label], float(epsilon))
            rel_name = src_path.relative_to(project_root / "data" / "frames")
            dst_path = out_dir / rel_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_path), attacked)
            split_lines.append(f"data/degraded/{folder_name}/{rel_name},{label}")

        split_path = splits_root / f"test_{folder_name}.txt"
        split_path.write_text("\n".join(split_lines) + "\n")
