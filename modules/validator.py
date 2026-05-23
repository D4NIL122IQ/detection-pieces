from __future__ import annotations

"""Validateur final de cercles détectés.

Après la détection Hough, ce module filtre les faux positifs en analysant
deux signaux complémentaires pour chaque cercle candidat :

1. **Score de bord (edge_score)** : gradient Sobel le long du périmètre.
   Un vrai bord de pièce produit un gradient fort et continu.
   Les faux cercles issus de motifs ou de textures ont un gradient faible/partiel.

2. **Score de métallicité (metallic_score)** : analyse de l'intérieur du cercle.
   Une pièce est un disque métallique = luminosité modérée (pas fond blanc, pas ombre noire)
   et saturation faible à modérée. L'intérieur doit être relativement homogène.

Seuils conservateurs : on préfère garder quelques faux positifs plutôt que de supprimer
des vraies pièces (rappel > précision dans les arbitrages).
"""

import cv2
import numpy as np

from modules.segmentation import DetectedCircle


# -- Seuils de rejet ---------------------------------------------------------
# Ces seuils sont intentionnellement bas pour ne pas nuire au rappel.
# On ne rejette que les cas clairement non-pièces.
_EDGE_SCORE_MIN = 0.20       # gradient périmètre trop faible → pas un bord de pièce
_METALLIC_SCORE_MIN = 0.28   # intérieur incohérent avec du métal
_COVERAGE_MIN = 0.55         # moins de 55% du cercle dans l'image → tronqué/hors-cadre


def _coverage_score(circle: DetectedCircle, image_shape: tuple[int, int]) -> float:
    """Proportion de la surface du cercle qui est dans l'image.

    Un cercle partiellement hors image peut être un artefact de bord.
    Score entre 0 (entièrement hors image) et 1 (entièrement dans l'image).
    """
    h, w = image_shape
    cx, cy, r = circle.x, circle.y, circle.radius
    if r <= 0:
        return 0.0

    # Boîte englobante clippée
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)

    # Surface théorique du cercle
    area_circle = np.pi * r * r

    # Approximation : fraction de la boîte englobante dans l'image
    box_w = (cx + r) - (cx - r)
    box_h = (cy + r) - (cy - r)
    box_area = box_w * box_h if box_w > 0 and box_h > 0 else 1.0
    clipped_area = (x2 - x1) * (y2 - y1)
    return float(np.clip(clipped_area / box_area, 0.0, 1.0))


def _edge_score(gray: np.ndarray, circle: DetectedCircle) -> float:
    """Mesure la force du gradient le long du périmètre.

    Retourne un score entre 0 (pas de bord) et 1 (bord net et continu).
    """
    h, w = gray.shape[:2]
    cx, cy, r = circle.x, circle.y, circle.radius
    if r < 5:
        return 0.0

    n_points = 72
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    px = (cx + r * np.cos(angles)).astype(int)
    py = (cy + r * np.sin(angles)).astype(int)

    valid = (px >= 1) & (px < w - 1) & (py >= 1) & (py < h - 1)
    if valid.sum() < n_points * 0.5:
        return 0.0

    px, py = px[valid], py[valid]

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_values = grad_mag[py, px]
    mean_edge = float(np.mean(edge_values))
    global_mean = float(np.mean(grad_mag)) + 1e-6

    return float(np.clip(mean_edge / (global_mean * 3.0), 0.0, 1.0))


def _metallic_score(image_bgr: np.ndarray, circle: DetectedCircle) -> float:
    """Mesure si l'intérieur du cercle ressemble à du métal.

    Critères :
    - Luminosité modérée (V entre 60 et 240) : pas fond blanc ni ombre noire
    - Saturation faible à modérée (S entre 10 et 180) : métal ≠ couleur vive
    - Homogénéité raisonnable (écart-type V pas trop élevé)

    Retourne un score entre 0 (pas métallique) et 1 (très métallique).
    """
    h_img, w_img = image_bgr.shape[:2]
    cx, cy, r = circle.x, circle.y, circle.radius
    r_int = max(1, int(r * 0.80))  # légèrement réduit pour éviter l'arrière-plan

    x1 = max(0, cx - r_int)
    y1 = max(0, cy - r_int)
    x2 = min(w_img, cx + r_int)
    y2 = min(h_img, cy + r_int)

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Masque circulaire
    lx, ly = cx - x1, cy - y1
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), r_int, 255, -1)

    if mask.sum() == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2][mask > 0].astype(float)
    s = hsv[:, :, 1][mask > 0].astype(float)

    if len(v) < 20:
        return 0.0

    # Fraction de pixels dans la plage "métal"
    metal_v = np.sum((v >= 50) & (v <= 245)) / len(v)
    metal_s = np.sum(s <= 190) / len(v)

    # Homogénéité : un fond blanc uniforme ou un arrière-plan a un std faible
    # mais une luminosité très haute. On pénalise les zones trop uniformes ET claires.
    v_mean = float(np.mean(v))
    v_std = float(np.std(v))

    # Fond blanc pur → V proche de 255, std faible
    is_white_background = (v_mean > 230) and (v_std < 20)
    # Fond noir pur → V proche de 0
    is_black_background = v_mean < 30

    if is_white_background or is_black_background:
        return 0.1

    # Score composite
    score = metal_v * 0.55 + metal_s * 0.30 + min(1.0, v_std / 30.0) * 0.15
    return float(np.clip(score, 0.0, 1.0))


def validate_coins(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
    *,
    min_edge: float = _EDGE_SCORE_MIN,
    min_metallic: float = _METALLIC_SCORE_MIN,
    min_coverage: float = _COVERAGE_MIN,
    keep_all_if_few: int = 1,
) -> list[DetectedCircle]:
    """Filtre les faux positifs parmi les cercles détectés.

    Pour chaque cercle, calcule trois scores indépendants et rejette le cercle
    si deux d'entre eux sont sous leur seuil (vote majoritaire laxiste).

    Paramètres :
        keep_all_if_few : si <= N cercles en entrée, ne rien rejeter (évite de
                          supprimer le seul cercle sur une image difficile).
    """
    if not circles:
        return []

    if len(circles) <= keep_all_if_few:
        return list(circles)

    h_img, w_img = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Pré-calcul CLAHE sur gris pour edge_score plus robuste
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    validated = []
    for circle in circles:
        e_score = _edge_score(gray_eq, circle)
        m_score = _metallic_score(image_bgr, circle)
        c_score = _coverage_score(circle, (h_img, w_img))

        # Compter les critères échoués
        failures = 0
        if e_score < min_edge:
            failures += 1
        if m_score < min_metallic:
            failures += 1
        if c_score < min_coverage:
            failures += 1

        # Rejeté seulement si au moins 2 critères échouent (vote 2/3)
        if failures < 2:
            validated.append(circle)

    # Sécurité : ne jamais tout rejeter — si on aurait tout rejeté, on garde le meilleur
    if not validated and circles:
        # Garder le cercle avec le meilleur score composite
        def _composite(c: DetectedCircle) -> float:
            return (
                _edge_score(gray_eq, c) +
                _metallic_score(image_bgr, c) +
                _coverage_score(c, (h_img, w_img))
            )
        validated = [max(circles, key=_composite)]

    return validated
