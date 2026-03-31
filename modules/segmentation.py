from __future__ import annotations

"""Fonctions de détection de pièces par transformée de Hough.

Ce module regroupe tout le pipeline de détection :
- redimensionnement pour travailler à une résolution stable ;
- amélioration locale du contraste ;
- prétraitement avant Hough ;
- détection des cercles ;
- suppression légère des doublons ;
- dessin des résultats.
"""

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
FALLBACK_PARAM2 = 28
FALLBACK_MIN_RADIUS_RATIO = 0.02
FALLBACK_MAX_RADIUS_RATIO = 0.35


@dataclass(frozen=True)
class DetectedCircle:
    """Représentation simple d'un cercle détecté dans l'image."""

    x: int
    y: int
    radius: int
    score: float | None = None


def resize_for_detection(image: np.ndarray, max_size: int = TAILLE_MAX) -> tuple[np.ndarray, float]:
    """Redimensionne l'image sans agrandir si elle est déjà assez petite.

    Returns:
        tuple[np.ndarray, float]:
            - l'image redimensionnée ;
            - le facteur d'échelle appliqué.
    """

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
    """Améliore localement le contraste via CLAHE sur la luminance LAB."""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_for_hough(image: np.ndarray) -> np.ndarray:
    """Prépare l'image pour Hough en produisant une image en niveaux de gris lissée."""

    normalized = apply_clahe_bgr(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, BLUR_MEDIAN)
    return cv2.GaussianBlur(median, (BLUR_GAUSS, BLUR_GAUSS), 0)


def _deduplicate_circles(circles: list[DetectedCircle]) -> list[DetectedCircle]:
    """Fusionne les détections redondantes produites par Hough.

    La transformée de Hough retourne parfois :
    - plusieurs fois le même cercle ;
    - un cercle intérieur et un cercle extérieur pour une même pièce ;
    - quelques rayons aberrants lorsqu'il y a déjà beaucoup de détections valides.
    """

    deduplicated: list[DetectedCircle] = []
    for circle in sorted(circles, key=lambda item: item.radius):
        keep = True
        for selected in deduplicated:
            center_distance = float(np.hypot(circle.x - selected.x, circle.y - selected.y))
            radius_gap = abs(circle.radius - selected.radius)
            larger = max(circle.radius, selected.radius)
            smaller = min(circle.radius, selected.radius)

            if center_distance < min(circle.radius, selected.radius) * 0.6 and radius_gap < max(
                4, min(circle.radius, selected.radius) * 0.35
            ):
                # Même centre et rayon proche : on considère que c'est le même cercle.
                keep = False
                break

            if center_distance < max(circle.radius, selected.radius) * 0.22 and circle.radius < selected.radius * 0.72:
                # Petit cercle très proche d'un plus grand : souvent un doublon intérieur.
                keep = False
                break

            # Remove nested circles where Hough detects both an inner and outer ring for one piece.
            if larger > smaller * 1.45 and center_distance < larger * 0.62:
                keep = False
                break

        if keep:
            deduplicated.append(circle)

    if len(deduplicated) >= 4:
        radii = np.array([circle.radius for circle in deduplicated], dtype=np.float32)
        median_radius = float(np.median(radii))
        max_reasonable_radius = median_radius * 1.85
        # Quand il y a déjà plusieurs pièces détectées, on supprime les très gros rayons
        # qui correspondent souvent à des faux positifs.
        deduplicated = [circle for circle in deduplicated if circle.radius <= max_reasonable_radius]

    return sorted(deduplicated, key=lambda item: (item.y, item.x))


def _run_hough(
    image: np.ndarray,
    dp: float,
    min_dist: int,
    param2: int,
    min_radius: int,
    max_radius: int,
) -> list[DetectedCircle]:
    """Exécute une passe Hough et convertit le résultat OpenCV en objets Python."""

    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=PARAM1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    detected = []
    for x, y, radius in np.round(circles[0]).astype(int):
        detected.append(DetectedCircle(x=int(x), y=int(y), radius=int(radius)))
    return detected


def detect_coins(image: np.ndarray) -> list[DetectedCircle]:
    """Détecte les pièces présentes dans une image couleur.

    La stratégie est volontairement simple :
    1. redimensionnement ;
    2. prétraitement ;
    3. passe Hough principale ;
    4. passe de secours plus permissive si rien n'a été trouvé ;
    5. remise à l'échelle et nettoyage des doublons.
    """

    resized, scale = resize_for_detection(image)
    prepared = preprocess_for_hough(resized)
    height, width = prepared.shape[:2]
    min_dim = min(height, width)

    min_radius = max(8, int(round(min_dim * RAYON_MIN_RATIO)))
    max_radius = max(min_radius + 2, int(round(min_dim * RAYON_MAX_RATIO)))
    min_dist = max(min_radius * 2, int(round(min_dim * MIN_DIST_RATIO)))

    detected = _run_hough(prepared, DP, min_dist, PARAM2, min_radius, max_radius)

    if not detected:
        # La seconde passe n'est lancée qu'en cas d'échec complet pour éviter
        # de dégrader les cas simples qui fonctionnent déjà bien.
        fallback_min_radius = max(8, int(round(min_dim * FALLBACK_MIN_RADIUS_RATIO)))
        fallback_max_radius = max(fallback_min_radius + 2, int(round(min_dim * FALLBACK_MAX_RADIUS_RATIO)))
        fallback_min_dist = max(20, int(round(min_dim * 0.06)))
        detected = _run_hough(
            prepared,
            1.2,
            fallback_min_dist,
            FALLBACK_PARAM2,
            fallback_min_radius,
            fallback_max_radius,
        )

    if not detected:
        return []

    scale_back = 1.0 / scale
    detected = [
        DetectedCircle(
            x=int(round(circle.x * scale_back)),
            y=int(round(circle.y * scale_back)),
            radius=int(round(circle.radius * scale_back)),
        )
        for circle in detected
    ]
    return _deduplicate_circles(detected)


def draw_circles(
    image: np.ndarray,
    predicted: list[DetectedCircle],
    ground_truth: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Dessine les cercles prédits et, si disponible, les annotations de référence."""

    canvas = image.copy()
    if ground_truth is not None:
        for x, y, radius in ground_truth:
            cv2.circle(canvas, (int(round(x)), int(round(y))), int(round(radius)), (255, 120, 0), 3)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (255, 120, 0), -1)

    for circle in predicted:
        cv2.circle(canvas, (circle.x, circle.y), circle.radius, (40, 220, 40), 3)
        cv2.circle(canvas, (circle.x, circle.y), 2, (40, 220, 40), -1)

    return canvas
