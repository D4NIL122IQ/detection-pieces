from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CircleAnnotation:
    label: str
    x: float
    y: float
    radius: float


def _circle_from_shape(shape: dict[str, Any]) -> CircleAnnotation | None:
    if shape.get("shape_type") != "circle":
        return None

    points = shape.get("points", [])
    if len(points) != 2:
        return None

    (x1, y1), (x2, y2) = points
    radius = math.hypot(x2 - x1, y2 - y1)
    return CircleAnnotation(
        label=str(shape.get("label", "unknown")),
        x=float(x1),
        y=float(y1),
        radius=float(radius),
    )


def load_labelme_annotation(annotation_path: str | Path) -> dict[str, Any]:
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    circles = []
    for shape in data.get("shapes", []):
        circle = _circle_from_shape(shape)
        if circle is not None:
            circles.append(circle)

    return {
        "path": annotation_path,
        "image_path": data.get("imagePath"),
        "image_data": data.get("imageData"),
        "image_width": data.get("imageWidth"),
        "image_height": data.get("imageHeight"),
        "circles": circles,
    }


def decode_labelme_image(image_data: str | None) -> np.ndarray | None:
    if not image_data:
        return None

    buffer = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image
