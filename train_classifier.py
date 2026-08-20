from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.model import build_cnn, resolve_accident_index, safe_torch_load, training_transform, validation_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the accident/non-accident CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/best_accident_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_split(root: Path, requested_name: str) -> Path:
    aliases = {
        "train": {"train", "training"},
        "val": {"val", "valid", "validation"},
        "test": {"test", "testing"},
    }[requested_name]

    candidates = [directory for directory in root.rglob("*") if directory.is_dir()]
    for directory in [root, *candidates]:
        if directory.name.lower().strip() in aliases:
            subdirectories = [item for item in directory.iterdir() if item.is_dir()]
            if len(subdirectories) >= 2:
                return directory
    raise FileNotFoundError(
        f"Could not locate the '{requested_name}' split inside {root}. "
        "Expected train/Accident, train/Non Accident, val/..., and test/..."
    )


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], list[str], dict[str, int]]:
    split_paths = {name: find_split(args.data_dir, name) for name in ("train", "val", "test")}
    datasets = {
        "train": ImageFolder(split_paths["train"], transform=training_transform(args.input_size)),
        "val": ImageFolder(split_paths["val"], transform=validation_transform(args.input_size)),
        "test": ImageFolder(split_paths["test"], transform=validation_transform(args.input_size)),
    }

    class_names = datasets["train"].classes
    for split_name, dataset in datasets.items():
        if dataset.classes != class_names:
            raise ValueError(
                f"Class ordering differs in {split_name}: {dataset.classes}; training classes: {class_names}"
            )

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
    sizes = {name: len(dataset) for name, dataset in datasets.items()}
    return loaders, class_names, sizes


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    progress = tqdm(loader, leave=False, desc="train" if training else "evaluate")
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    average_loss = running_loss / max(1, len(loader.dataset))
    accuracy = accuracy_score(labels_all, predictions_all)
    macro_f1 = f1_score(labels_all, predictions_all, average="macro", zero_division=0)
    return float(average_loss), float(accuracy), float(macro_f1)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders, class_names, sizes = make_loaders(args)
    accident_index = resolve_accident_index(class_names)
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {sizes}")
    print(f"Accident class index: {accident_index}")

    model = build_cnn(num_classes=len(class_names), pretrained=True).to(device)
    set_backbone_trainable(model, False)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_f1 = -1.0
    stale_epochs = 0
    history: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = AdamW(model.parameters(), lr=args.learning_rate * 0.2, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            print("Backbone unfrozen for fine-tuning.")

        train_loss, train_accuracy, train_f1 = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_accuracy, val_f1 = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "accident_index": accident_index,
                    "input_size": args.input_size,
                    "best_val_macro_f1": best_f1,
                    "training_history": history,
                },
                args.output,
            )
            print(f"Saved improved checkpoint to {args.output}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping activated.")
                break

    checkpoint = safe_torch_load(args.output, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            logits = model(images.to(device))
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())

    test_report = {
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels_all, predictions_all).tolist(),
        "classification_report": classification_report(
            labels_all,
            predictions_all,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    print(f"Test metrics saved to {metrics_path}")
    print(json.dumps(test_report, indent=2))


if __name__ == "__main__":
    main()
