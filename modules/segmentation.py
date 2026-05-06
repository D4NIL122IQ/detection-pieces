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


def apply_clahe_bgr(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Améliore localement le contraste via CLAHE sur la luminance LAB."""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_for_hough(image: np.ndarray) -> np.ndarray:
    """Prépare l'image pour Hough en produisant une image en niveaux de gris lissée."""

    normalized = apply_clahe_bgr(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, BLUR_MEDIAN)
    return cv2.GaussianBlur(median, (BLUR_GAUSS, BLUR_GAUSS), 0)


def _edge_score(gray: np.ndarray, circle: DetectedCircle) -> float:
    """Mesure la force du gradient le long du périmètre du cercle détecté.

    Un vrai bord de pièce a un gradient fort et continu sur son contour.
    Les faux cercles (motifs, reliefs) ont un gradient faible ou partiel.
    Retourne un score entre 0 (pas de bord) et 1 (bord net continu).
    """
    h, w = gray.shape[:2]
    cx, cy, r = circle.x, circle.y, circle.radius
    if r < 5:
        return 0.0

    # Échantillonner 72 points le long du périmètre (tous les 5°)
    n_points = 72
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    px = (cx + r * np.cos(angles)).astype(int)
    py = (cy + r * np.sin(angles)).astype(int)

    # Garder les points dans l'image
    valid = (px >= 1) & (px < w - 1) & (py >= 1) & (py < h - 1)
    if valid.sum() < n_points * 0.5:
        return 0.0

    px, py = px[valid], py[valid]

    # Gradient (Sobel) aux points du périmètre
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Magnitude moyenne du gradient sur le périmètre
    edge_values = grad_mag[py, px]
    mean_edge = float(np.mean(edge_values))

    # Normaliser par le gradient moyen global (pour être invariant au contraste)
    global_mean = float(np.mean(grad_mag)) + 1e-6
    score = mean_edge / (global_mean * 3.0)

    return float(np.clip(score, 0.0, 1.0))


def _deduplicate_circles(
    circles: list[DetectedCircle],
    image_shape: tuple[int, int] | None = None,
) -> list[DetectedCircle]:
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

    if image_shape is not None and deduplicated:
        height, width = image_shape
        max_dim = max(height, width)
        sorted_by_radius = sorted(deduplicated, key=lambda item: item.radius, reverse=True)
        largest = sorted_by_radius[0]
        second_radius = sorted_by_radius[1].radius if len(sorted_by_radius) >= 2 else 0

        is_dominant_large_circle = (
            largest.radius >= max_dim * 0.22
            and (second_radius == 0 or largest.radius >= second_radius * 1.55)
        )
        if is_dominant_large_circle:
            filtered = [largest]
            for circle in deduplicated:
                if circle is largest:
                    continue
                center_distance = float(np.hypot(circle.x - largest.x, circle.y - largest.y))

                # Gros plan sur une seule pièce : les petits cercles internes générés
                # par les motifs de surface sont rejetés.
                if center_distance < largest.radius * 0.72 and circle.radius < largest.radius * 0.78:
                    continue

                filtered.append(circle)
            deduplicated = filtered

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


def _detect_closeup_coin(gray: np.ndarray) -> DetectedCircle | None:
    """Détecte une pièce en gros plan via contours + ajustement de cercle.

    Utilisé quand la pièce occupe une grande partie de l'image et que
    Hough ne peut pas la détecter (rayon trop grand).
    Cherche le plus grand contour circulaire dans l'image.
    """
    h, w = gray.shape[:2]
    min_dim = min(h, w)

    blurred = cv2.GaussianBlur(gray, (15, 15), 3)
    edges = cv2.Canny(blurred, 30, 80)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_circle = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min_dim * 0.15) ** 2 * np.pi:
            continue

        # Ajuster un cercle minimum englobant
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_dim * 0.25:
            continue

        # Vérifier la circularité : aire contour vs aire cercle
        circle_area = np.pi * radius ** 2
        circularity = area / circle_area if circle_area > 0 else 0
        if circularity < 0.4:
            continue

        if area > best_area:
            best_area = area
            best_circle = DetectedCircle(
                x=int(round(cx)),
                y=int(round(cy)),
                radius=int(round(radius)),
            )

    return best_circle


def detect_coins(image: np.ndarray) -> list[DetectedCircle]:
    """Détecte les pièces présentes dans une image couleur.

    La stratégie est volontairement simple :
    1. redimensionnement ;
    2. prétraitement ;
    3. passe Hough principale ;
    4. passe de secours plus permissive si rien n'a été trouvé ;
    5. détection gros plan si beaucoup de cercles sans dominant ;
    6. remise à l'échelle et nettoyage des doublons.
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

    # Détection de gros plan : si beaucoup de cercles sont tous concentrés
    # dans une même zone, c'est probablement un gros plan d'une seule pièce
    # dont les motifs internes créent des faux cercles.
    if len(detected) >= 6:
        xs = np.array([c.x for c in detected], dtype=float)
        ys = np.array([c.y for c in detected], dtype=float)
        radii = np.array([c.radius for c in detected], dtype=float)

        # Centre de masse des détections
        cx_mean, cy_mean = float(xs.mean()), float(ys.mean())

        # Distance max d'un centre de cercle au barycentre
        dists = np.sqrt((xs - cx_mean) ** 2 + (ys - cy_mean) ** 2)
        max_dist = float(dists.max())

        # Si tous les cercles sont concentrés dans une zone < 25% de l'image
        # et qu'aucun cercle n'est dominant (rayon max < 25% image),
        # c'est un gros plan → garder un seul cercle englobant
        if max_dist < min_dim * 0.25 and float(radii.max()) < min_dim * 0.25:
            enclosing_r = max_dist + float(radii[int(np.argmax(dists))])
            detected = [DetectedCircle(
                x=int(round(cx_mean)),
                y=int(round(cy_mean)),
                radius=int(round(enclosing_r)),
            )]

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
    return _deduplicate_circles(detected, image.shape[:2])


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
