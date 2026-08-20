from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .geometry import Box, box_center, box_diagonal, euclidean, iou, mean_point
from .trajectory import TrajectoryStore


@dataclass(slots=True)
class TrackedVehicle:
    track_id: int
    box: Box
    label: str
    confidence: float


@dataclass(slots=True)
class AccidentEvent:
    frame_index: int
    timestamp_seconds: float
    confidence: float
    involved_vehicle_ids: list[int]
    collision_point: tuple[int, int]
    pair_score: float
    pair_selection_reliable: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["collision_point"] = list(self.collision_point)
        return result


class TemporalAccidentGate:
    def __init__(
        self,
        window: int,
        frame_positive_threshold: float,
        on_threshold: float,
        off_threshold: float,
        minimum_positive_predictions: int,
    ) -> None:
        self.probabilities: deque[float] = deque(maxlen=window)
        self.frame_positive_threshold = frame_positive_threshold
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.minimum_positive_predictions = minimum_positive_predictions
        self.active = False

    def update(self, probability: float) -> tuple[float, bool, bool]:
        self.probabilities.append(float(probability))
        smoothed = float(np.mean(self.probabilities))
        positive_count = sum(value >= self.frame_positive_threshold for value in self.probabilities)
        newly_triggered = False

        if not self.active and len(self.probabilities) == self.probabilities.maxlen:
            if smoothed >= self.on_threshold and positive_count >= self.minimum_positive_predictions:
                self.active = True
                newly_triggered = True
        elif self.active and smoothed <= self.off_threshold:
            self.active = False

        return smoothed, self.active, newly_triggered


def select_involved_pair(
    vehicles: Iterable[TrackedVehicle],
    trajectories: TrajectoryStore,
    minimum_pair_score: float,
) -> tuple[list[int], tuple[int, int], float, bool]:
    vehicle_list = list(vehicles)
    if len(vehicle_list) < 2:
        if vehicle_list:
            vehicle = vehicle_list[0]
            return [vehicle.track_id], mean_point([box_center(vehicle.box)]), 0.0, False
        return [], (0, 0), 0.0, False

    best_pair: tuple[TrackedVehicle, TrackedVehicle] | None = None
    best_score = -1.0

    for first, second in combinations(vehicle_list, 2):
        overlap = iou(first.box, second.box)
        center_distance = euclidean(box_center(first.box), box_center(second.box))
        reference_scale = max(1.0, (box_diagonal(first.box) + box_diagonal(second.box)) / 2.0)
        proximity = float(np.clip(1.0 - center_distance / (1.75 * reference_scale), 0.0, 1.0))
        deceleration = (
            trajectories.deceleration_score(first.track_id) + trajectories.deceleration_score(second.track_id)
        ) / 2.0
        heading_change = (
            trajectories.heading_change_score(first.track_id)
            + trajectories.heading_change_score(second.track_id)
        ) / 2.0

        score = 0.35 * overlap + 0.35 * proximity + 0.20 * deceleration + 0.10 * heading_change
        if score > best_score:
            best_score = score
            best_pair = (first, second)

    assert best_pair is not None
    first, second = best_pair
    collision_point = mean_point([box_center(first.box), box_center(second.box)])
    reliable = best_score >= minimum_pair_score
    return [first.track_id, second.track_id], collision_point, float(best_score), reliable
