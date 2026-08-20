from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ClassifierConfig:
    checkpoint: str
    input_size: int = 224
    classify_every_n_frames: int = 2
    smoothing_window: int = 12
    frame_positive_threshold: float = 0.60
    accident_on_threshold: float = 0.70
    accident_off_threshold: float = 0.45
    minimum_positive_predictions: int = 5


@dataclass(slots=True)
class TrackingConfig:
    detector_model: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    confidence: float = 0.30
    iou_threshold: float = 0.50
    vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)
    history_length: int = 300


@dataclass(slots=True)
class IncidentConfig:
    cooldown_frames: int = 150
    minimum_pair_score: float = 0.20
    path_thickness: int = 3
    show_all_tracks: bool = True


@dataclass(slots=True)
class ProjectConfig:
    output_dir: str = "outputs"


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig
    classifier: ClassifierConfig
    tracking: TrackingConfig
    incident: IncidentConfig


def _resolve_path(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    root = config_path.parent
    project_raw = raw.get("project", {})
    classifier_raw = raw.get("classifier", {})
    tracking_raw = raw.get("tracking", {})
    incident_raw = raw.get("incident", {})

    project_raw["output_dir"] = _resolve_path(project_raw.get("output_dir", "outputs"), root)
    classifier_raw["checkpoint"] = _resolve_path(
        classifier_raw.get("checkpoint", "artifacts/best_accident_cnn.pt"), root
    )
    tracking_raw["vehicle_classes"] = tuple(tracking_raw.get("vehicle_classes", [2, 3, 5, 7]))

    return AppConfig(
        project=ProjectConfig(**project_raw),
        classifier=ClassifierConfig(**classifier_raw),
        tracking=TrackingConfig(**tracking_raw),
        incident=IncidentConfig(**incident_raw),
    )
