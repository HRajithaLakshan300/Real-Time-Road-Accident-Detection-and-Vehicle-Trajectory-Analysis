from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm

from src.model import validation_transform
from train_classifier import find_split


class MobileNetFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.avgpool(features)
        return torch.flatten(features, 1)


def extract_features(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    extractor.eval()
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc="Extracting CNN features"):
            batch_features = extractor(images.to(device)).cpu().numpy()
            features.append(batch_features)
            labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pattern-recognition baseline: pretrained CNN features followed by an SVM classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cnn_feature_svm.joblib"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ImageFolder(find_split(args.data_dir, "train"), transform=validation_transform(args.input_size))
    val_dataset = ImageFolder(find_split(args.data_dir, "val"), transform=validation_transform(args.input_size))
    test_dataset = ImageFolder(find_split(args.data_dir, "test"), transform=validation_transform(args.input_size))

    if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
        raise ValueError("Class folders must be identical in train, val, and test splits.")

    loaders = [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        for dataset in (train_dataset, val_dataset, test_dataset)
    ]

    extractor = MobileNetFeatureExtractor().to(device)
    train_x, train_y = extract_features(extractor, loaders[0], device)
    val_x, val_y = extract_features(extractor, loaders[1], device)
    test_x, test_y = extract_features(extractor, loaders[2], device)

    fit_x = np.concatenate([train_x, val_x])
    fit_y = np.concatenate([train_y, val_y])

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]
    )
    parameter_grid = {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", 0.01, 0.001],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, parameter_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1)
    search.fit(fit_x, fit_y)

    predictions = search.predict(test_x)
    report = {
        "best_parameters": search.best_params_,
        "best_cv_macro_f1": search.best_score_,
        "test_accuracy": accuracy_score(test_y, predictions),
        "test_macro_f1": f1_score(test_y, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
        "classification_report": classification_report(
            test_y,
            predictions,
            target_names=train_dataset.classes,
            output_dict=True,
            zero_division=0,
        ),
        "class_names": train_dataset.classes,
        "input_size": args.input_size,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": search.best_estimator_,
            "class_names": train_dataset.classes,
            "input_size": args.input_size,
        },
        args.output,
    )
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
