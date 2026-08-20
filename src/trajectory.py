from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import Box, Point, bottom_center, euclidean


@dataclass(slots=True)
class TrackPoint:
    frame_index: int
    point: Point
    box: Box


class TrajectoryStore:
    def __init__(self, max_length: int = 300) -> None:
        self.histories: dict[int, deque[TrackPoint]] = defaultdict(lambda: deque(maxlen=max_length))
        self.latest_boxes: dict[int, Box] = {}
        self.latest_labels: dict[int, str] = {}

    def update(self, track_id: int, box: Box, frame_index: int, label: str) -> None:
        point = bottom_center(box)
        history = self.histories[track_id]
        if history and history[-1].frame_index == frame_index:
            return
        history.append(TrackPoint(frame_index=frame_index, point=point, box=box))
        self.latest_boxes[track_id] = box
        self.latest_labels[track_id] = label

    def recent_speed(self, track_id: int, lookback: int = 3) -> float:
        history = self.histories.get(track_id)
        if not history or len(history) < 2:
            return 0.0
        points = list(history)
        start = points[max(0, len(points) - 1 - lookback)]
        end = points[-1]
        frames = max(1, end.frame_index - start.frame_index)
        return euclidean(start.point, end.point) / frames

    def deceleration_score(self, track_id: int, segment: int = 3) -> float:
        history = self.histories.get(track_id)
        if not history or len(history) < (segment * 2 + 1):
            return 0.0
        points = list(history)
        a0 = points[-(segment * 2 + 1)]
        a1 = points[-(segment + 1)]
        b0 = points[-(segment + 1)]
        b1 = points[-1]
        speed_before = euclidean(a0.point, a1.point) / max(1, a1.frame_index - a0.frame_index)
        speed_after = euclidean(b0.point, b1.point) / max(1, b1.frame_index - b0.frame_index)
        if speed_before < 1e-6:
            return 0.0
        return float(np.clip((speed_before - speed_after) / speed_before, 0.0, 1.0))

    def heading_change_score(self, track_id: int, segment: int = 3) -> float:
        history = self.histories.get(track_id)
        if not history or len(history) < (segment * 2 + 1):
            return 0.0
        points = list(history)
        p0 = points[-(segment * 2 + 1)].point
        p1 = points[-(segment + 1)].point
        p2 = points[-1].point
        angle_1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        angle_2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        difference = abs((angle_2 - angle_1 + math.pi) % (2 * math.pi) - math.pi)
        return float(np.clip(difference / math.pi, 0.0, 1.0))

    def draw(
        self,
        frame: np.ndarray,
        event_frame: int | None,
        involved_ids: set[int],
        thickness: int = 3,
        show_all_tracks: bool = True,
    ) -> None:
        for track_id, history in self.histories.items():
            if len(history) < 2:
                continue

            points = list(history)
            if track_id in involved_ids and event_frame is not None:
                before = [item.point for item in points if item.frame_index <= event_frame]
                after = [item.point for item in points if item.frame_index > event_frame]
                self._draw_polyline(frame, before, (0, 255, 0), thickness)
                self._draw_polyline(frame, after, (0, 0, 255), thickness)
            elif show_all_tracks:
                self._draw_polyline(frame, [item.point for item in points], (255, 200, 0), 2)

    @staticmethod
    def _draw_polyline(frame: np.ndarray, points: list[Point], color: tuple[int, int, int], thickness: int) -> None:
        if len(points) < 2:
            return
        array = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [array], isClosed=False, color=color, thickness=thickness)
