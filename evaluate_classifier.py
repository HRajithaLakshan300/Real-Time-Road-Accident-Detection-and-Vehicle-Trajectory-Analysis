from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model import build_cnn, safe_torch_load, validation_transform
from train_classifier import find_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_accident_cnn.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = safe_torch_load(args.checkpoint, device)
    class_names = list(checkpoint["class_names"])
    input_size = int(checkpoint.get("input_size", 224))

    test_path = find_split(args.data_dir, "test")
    dataset = ImageFolder(test_path, transform=validation_transform(input_size))
    if dataset.classes != class_names:
        raise ValueError(f"Checkpoint classes {class_names} do not match test classes {dataset.classes}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_cnn(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions.tolist())

    report = classification_report(
        labels_all,
        predictions_all,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels_all, predictions_all)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(values_format="d")
    plt.title("Accident CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
