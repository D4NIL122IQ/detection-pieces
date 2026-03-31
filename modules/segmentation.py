from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


TAILLE_MAX = 800
BLUR_MEDIAN = 15
BLUR_GAUSS = 11
PARAM1 = 80
PARAM2 = 40
DP = 1.2
RAYON_MIN_RATIO = 0.03
RAYON_MAX_RATIO = 0.30
MIN_DIST_RATIO = 0.08


@dataclass(frozen=True)
class DetectedCircle:
    x: int
    y: int
    radius: int
    score: float | None = None


def resize_for_detection(image: np.ndarray, max_size: int = TAILLE_MAX) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale == 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def apply_clahe_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_for_hough(image: np.ndarray) -> np.ndarray:
    normalized = apply_clahe_bgr(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, BLUR_MEDIAN)
    return cv2.GaussianBlur(median, (BLUR_GAUSS, BLUR_GAUSS), 0)


def _deduplicate_circles(circles: list[DetectedCircle]) -> list[DetectedCircle]:
    deduplicated: list[DetectedCircle] = []
    for circle in sorted(circles, key=lambda item: item.radius, reverse=True):
        keep = True
        for selected in deduplicated:
            center_distance = float(np.hypot(circle.x - selected.x, circle.y - selected.y))
            radius_gap = abs(circle.radius - selected.radius)
            if center_distance < min(circle.radius, selected.radius) * 0.6 and radius_gap < max(
                4, min(circle.radius, selected.radius) * 0.35
            ):
                keep = False
                break
        if keep:
            deduplicated.append(circle)

    return sorted(deduplicated, key=lambda item: (item.y, item.x))


def detect_coins(image: np.ndarray) -> list[DetectedCircle]:
    resized, scale = resize_for_detection(image)
    prepared = preprocess_for_hough(resized)
    height, width = prepared.shape[:2]
    min_dim = min(height, width)

    min_radius = max(8, int(round(min_dim * RAYON_MIN_RATIO)))
    max_radius = max(min_radius + 2, int(round(min_dim * RAYON_MAX_RATIO)))
    min_dist = max(min_radius * 2, int(round(min_dim * MIN_DIST_RATIO)))

    circles = cv2.HoughCircles(
        prepared,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=min_dist,
        param1=PARAM1,
        param2=PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    scale_back = 1.0 / scale
    detected = []
    for x, y, radius in np.round(circles[0]).astype(int):
        detected.append(
            DetectedCircle(
                x=int(round(x * scale_back)),
                y=int(round(y * scale_back)),
                radius=int(round(radius * scale_back)),
            )
        )

    return _deduplicate_circles(detected)


def draw_circles(
    image: np.ndarray,
    predicted: list[DetectedCircle],
    ground_truth: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    if ground_truth is not None:
        for x, y, radius in ground_truth:
            cv2.circle(canvas, (int(round(x)), int(round(y))), int(round(radius)), (255, 120, 0), 3)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (255, 120, 0), -1)

    for circle in predicted:
        cv2.circle(canvas, (circle.x, circle.y), circle.radius, (40, 220, 40), 3)
        cv2.circle(canvas, (circle.x, circle.y), 2, (40, 220, 40), -1)

    return canvas
