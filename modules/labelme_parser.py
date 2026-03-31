from __future__ import annotations

"""Lecture et conversion des annotations LabelMe.

Le dataset fournit des annotations de type cercle au format JSON LabelMe.
Ce module extrait les informations utiles et peut aussi reconstruire
l'image embarquée dans le champ ``imageData``.
"""

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
    """Cercle annoté dans LabelMe."""

    label: str
    x: float
    y: float
    radius: float


def _circle_from_shape(shape: dict[str, Any]) -> CircleAnnotation | None:
    """Convertit une forme LabelMe en cercle si son type est compatible."""

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
    """Charge un fichier LabelMe et ne conserve que les informations utiles au projet."""

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
    """Décode l'image embarquée en base64 dans une annotation LabelMe."""

    if not image_data:
        return None

    buffer = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image
