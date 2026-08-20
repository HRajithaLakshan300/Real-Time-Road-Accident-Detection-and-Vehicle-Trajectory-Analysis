from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from .config import AppConfig
from .incident import AccidentEvent, TemporalAccidentGate, TrackedVehicle, select_involved_pair
from .model import AccidentFrameClassifier
from .trajectory import TrajectoryStore

ProgressCallback = Callable[[int, int | None], None]


class AccidentTrajectoryPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        checkpoint = Path(config.classifier.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {checkpoint}. Train it first with train_classifier.py."
            )

        self.classifier = AccidentFrameClassifier(checkpoint)
        self.detector = YOLO(config.tracking.detector_model)
        self.trajectories = TrajectoryStore(config.tracking.history_length)
        self.gate = TemporalAccidentGate(
            window=config.classifier.smoothing_window,
            frame_positive_threshold=config.classifier.frame_positive_threshold,
            on_threshold=config.classifier.accident_on_threshold,
            off_threshold=config.classifier.accident_off_threshold,
            minimum_positive_predictions=config.classifier.minimum_positive_predictions,
        )
        self.events: list[AccidentEvent] = []
        self.last_event_frame = -10**9
        self.latest_smoothed_probability = 0.0

    def process(
        self,
        source: str | int,
        output_video: str | Path,
        report_path: str | Path,
        display: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 1.0 or np.isnan(fps):
            fps = 25.0
        total_frames_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None

        output_video = Path(output_video)
        report_path = Path(report_path)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        writer: cv2.VideoWriter | None = None
        frame_index = 0
        last_raw_probability = 0.0

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {output_video}")

                vehicles = self._track_vehicles(frame, frame_index)

                if frame_index % self.config.classifier.classify_every_n_frames == 0:
                    last_raw_probability = self.classifier.predict_probability(frame)
                    smoothed, _active, triggered = self.gate.update(last_raw_probability)
                    self.latest_smoothed_probability = smoothed
                    if triggered and frame_index - self.last_event_frame >= self.config.incident.cooldown_frames:
                        ids, collision_point, pair_score, reliable = select_involved_pair(
                            vehicles,
                            self.trajectories,
                            self.config.incident.minimum_pair_score,
                        )
                        event = AccidentEvent(
                            frame_index=frame_index,
                            timestamp_seconds=frame_index / fps,
                            confidence=smoothed,
                            involved_vehicle_ids=ids,
                            collision_point=collision_point,
                            pair_score=pair_score,
                            pair_selection_reliable=reliable,
                        )
                        self.events.append(event)
                        self.last_event_frame = frame_index

                annotated = self._annotate(frame.copy(), vehicles, frame_index, last_raw_probability)
                writer.write(annotated)

                if display:
                    cv2.imshow("Accident Detection and Vehicle Trajectories", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if progress_callback is not None:
                    progress_callback(frame_index + 1, total_frames)
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        report = {
            "source": str(source),
            "output_video": str(output_video),
            "frames_processed": frame_index,
            "fps": fps,
            "events": [event.to_dict() for event in self.events],
            "limitations": [
                "The CNN was trained on still images, so temporal smoothing is used as a baseline.",
                "Involved vehicle IDs are estimated using proximity, overlap, deceleration, and heading change.",
                "Pixel trajectories are not real-world metres unless camera calibration is added.",
            ],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _track_vehicles(self, frame: np.ndarray, frame_index: int) -> list[TrackedVehicle]:
        results = self.detector.track(
            frame,
            persist=True,
            tracker=self.config.tracking.tracker,
            conf=self.config.tracking.confidence,
            iou=self.config.tracking.iou_threshold,
            classes=list(self.config.tracking.vehicle_classes),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()

        vehicles: list[TrackedVehicle] = []
        for box_values, track_id, class_id, confidence in zip(xyxy, ids, classes, confidences):
            box = tuple(float(value) for value in box_values)
            label = str(result.names.get(int(class_id), class_id))
            vehicle = TrackedVehicle(
                track_id=int(track_id),
                box=box,
                label=label,
                confidence=float(confidence),
            )
            vehicles.append(vehicle)
            self.trajectories.update(vehicle.track_id, vehicle.box, frame_index, vehicle.label)

        return vehicles

    def _annotate(
        self,
        frame: np.ndarray,
        vehicles: list[TrackedVehicle],
        frame_index: int,
        raw_probability: float,
    ) -> np.ndarray:
        latest_event = self.events[-1] if self.events else None
        involved_ids = set(latest_event.involved_vehicle_ids) if latest_event else set()
        event_frame = latest_event.frame_index if latest_event else None

        self.trajectories.draw(
            frame,
            event_frame=event_frame,
            involved_ids=involved_ids,
            thickness=self.config.incident.path_thickness,
            show_all_tracks=self.config.incident.show_all_tracks,
        )

        for vehicle in vehicles:
            x1, y1, x2, y2 = (int(value) for value in vehicle.box)
            is_involved = vehicle.track_id in involved_ids
            color = (0, 0, 255) if is_involved else (255, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{vehicle.label} ID:{vehicle.track_id} {vehicle.confidence:.2f}"
            cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        status = "ACCIDENT" if self.gate.active else "NORMAL"
        status_color = (0, 0, 255) if self.gate.active else (0, 200, 0)
        cv2.rectangle(frame, (10, 10), (390, 105), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
        cv2.putText(
            frame,
            f"CNN raw: {raw_probability:.2f}  smoothed: {self.latest_smoothed_probability:.2f}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(frame, f"Frame: {frame_index}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if latest_event is not None:
            cx, cy = latest_event.collision_point
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 4)
            reliability = "reliable" if latest_event.pair_selection_reliable else "uncertain"
            event_text = (
                f"Event IDs {latest_event.involved_vehicle_ids} | pair {latest_event.pair_score:.2f} ({reliability})"
            )
            cv2.putText(frame, event_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
