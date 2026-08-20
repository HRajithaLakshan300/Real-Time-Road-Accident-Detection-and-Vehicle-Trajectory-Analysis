from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_cnn(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def validation_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def training_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def safe_torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_accident_index(class_names: list[str]) -> int:
    normalized = [name.lower().replace("_", " ").replace("-", " ").strip() for name in class_names]

    for index, name in enumerate(normalized):
        if name in {"accident", "accidents", "crash", "collision"}:
            return index

    for index, name in enumerate(normalized):
        is_negative = any(token in name for token in ("non accident", "no accident", "normal", "safe"))
        if ("accident" in name or "crash" in name or "collision" in name) and not is_negative:
            return index

    raise ValueError(
        "Could not infer the accident class from checkpoint classes: "
        f"{class_names}. Rename the positive folder to 'Accident'."
    )


class AccidentFrameClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = safe_torch_load(checkpoint_path, self.device)
        self.class_names = list(checkpoint["class_names"])
        self.input_size = int(checkpoint.get("input_size", 224))
        self.accident_index = int(checkpoint.get("accident_index", resolve_accident_index(self.class_names)))

        self.model = build_cnn(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = validation_transform(self.input_size)

    @torch.inference_mode()
    def predict_probability(self, bgr_frame: np.ndarray) -> float:
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("An empty frame was supplied to the accident classifier.")

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        return float(probabilities[self.accident_index].item())
