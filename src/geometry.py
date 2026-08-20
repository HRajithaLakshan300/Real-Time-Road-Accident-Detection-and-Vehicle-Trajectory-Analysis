from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Box = tuple[float, float, float, float]
Point = tuple[int, int]


def bottom_center(box: Box) -> Point:
    x1, _y1, x2, y2 = box
    return int((x1 + x2) / 2.0), int(y2)


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def euclidean(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def box_diagonal(box: Box) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def mean_point(points: Iterable[tuple[float, float]]) -> Point:
    array = np.asarray(list(points), dtype=np.float32)
    if array.size == 0:
        return 0, 0
    average = array.mean(axis=0)
    return int(average[0]), int(average[1])
