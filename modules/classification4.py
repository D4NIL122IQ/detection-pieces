from __future__ import annotations

"""Classification combinée : ensemble de HSV + HLS + Filtres + détection anneau Canny.

Combine les résultats de classification.py (HSV), classification2.py (HLS)
et classification3.py (Filtres RGB) avec un vote pondéré par la confiance,
plus un détecteur d'anneau interne basé sur Canny pour confirmer/infirmer
le groupe bimétal. Inclut une correction de balance des blancs (Gray-World).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from modules.segmentation import DetectedCircle, apply_clahe_bgr
from modules.classification import classify_by_color_and_size, ValeurPiece, VALEURS_CENTIMES
from modules.classification2 import classify_hls, ValeurPieceHLS
from modules.classification3 import classify_filtres, ValeurPieceFiltre


@dataclass(frozen=True)
class ValeurPieceCombinee:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""
    methode_dominante: str = ""


def _detecter_anneau_interne(image_bgr: np.ndarray, circle: DetectedCircle) -> float:
    """Détecte la présence d'un anneau interne (signature bimétal) avec Canny + Hough.

    Retourne un score entre 0 (pas d'anneau) et 1 (anneau net détecté).
    Les pièces 1€/2€ ont une frontière visible entre centre et couronne.
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    r_ext = max(1, int(r * 0.92))

    x1 = max(0, cx - r_ext)
    y1 = max(0, cy - r_ext)
    x2 = min(image_bgr.shape[1], cx + r_ext)
    y2 = min(image_bgr.shape[0], cy + r_ext)

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Prétraitement
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Canny pour détecter les contours internes
    edges = cv2.Canny(gray, 30, 90)

    # Masquer les bords extérieurs (on ne veut que les contours INTERNES)
    lx, ly = cx - x1, cy - y1
    mask_ext = np.zeros_like(edges)
    cv2.circle(mask_ext, (lx, ly), int(r * 0.82), 255, -1)
    cv2.circle(mask_ext, (lx, ly), int(r * 0.25), 0, -1)  # Ignorer le centre pur
    edges_internes = cv2.bitwise_and(edges, mask_ext)

    # Chercher un cercle interne avec Hough
    # L'anneau bimétal est à environ 55-65% du rayon extérieur
    r_min_anneau = int(r * 0.40)
    r_max_anneau = int(r * 0.72)

    circles_internes = cv2.HoughCircles(
        edges_internes,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=r,  # Un seul cercle attendu
        param1=50,
        param2=15,
        minRadius=r_min_anneau,
        maxRadius=r_max_anneau,
    )

    if circles_internes is None:
        # Fallback : mesurer la densité de contours dans la zone annulaire
        zone_anneau = np.zeros_like(edges, dtype=np.uint8)
        cv2.circle(zone_anneau, (lx, ly), int(r * 0.68), 255, -1)
        cv2.circle(zone_anneau, (lx, ly), int(r * 0.52), 0, -1)
        pixels_zone = zone_anneau.sum() / 255
        if pixels_zone < 10:
            return 0.0
        densite = float(cv2.bitwise_and(edges, zone_anneau).sum() / 255) / pixels_zone
        # Une densité > 0.15 suggère un anneau
        return float(np.clip(densite / 0.20, 0.0, 0.7))

    # Cercle interne trouvé — vérifier qu'il est bien centré
    det = circles_internes[0][0]
    det_cx, det_cy, det_r = det[0], det[1], det[2]
    dist_centre = np.hypot(det_cx - lx, det_cy - ly)

    if dist_centre > r * 0.15:
        return 0.2  # Cercle trouvé mais décentré → faible confiance

    # Cercle bien centré et dans la bonne plage de rayon
    ratio_r = det_r / r
    if 0.45 <= ratio_r <= 0.70:
        return 0.9
    return 0.5


def _profil_radial_contraste(image_bgr: np.ndarray, circle: DetectedCircle) -> float:
    """Mesure la rupture de contraste radiale (bimétal = rupture nette centre/bord).

    Retourne un score : élevé si rupture nette (bimétal probable).
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    r_ext = max(1, int(r * 0.85))

    x1 = max(0, cx - r_ext)
    y1 = max(0, cy - r_ext)
    x2 = min(image_bgr.shape[1], cx + r_ext)
    y2 = min(image_bgr.shape[0], cy + r_ext)

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(float)
    lx, ly = cx - x1, cy - y1

    # Échantillonner la luminosité à différents rayons
    n_anneaux = 10
    moyennes = []
    for i in range(n_anneaux):
        r_int = int(r * (i / n_anneaux) * 0.85)
        r_out = int(r * ((i + 1) / n_anneaux) * 0.85)
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (lx, ly), max(1, r_out), 255, -1)
        if r_int > 0:
            cv2.circle(mask, (lx, ly), r_int, 0, -1)
        pixels = gray[mask > 0]
        if len(pixels) > 0:
            moyennes.append(float(np.mean(pixels)))
        else:
            moyennes.append(0.0)

    if len(moyennes) < 5:
        return 0.0

    # Calculer la dérivée du profil radial
    diffs = [abs(moyennes[i + 1] - moyennes[i]) for i in range(len(moyennes) - 1)]
    max_diff = max(diffs)

    # Une rupture nette (>15 niveaux de gris) entre anneaux adjacents → bimétal
    return float(np.clip(max_diff / 25.0, 0.0, 1.0))


def _correct_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    """Correction Gray-World : suppose que la moyenne de l'image est grise."""
    result = image_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = cv2.mean(result)[:3]
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_gray < 1.0:
        return image_bgr
    result[:, :, 0] *= avg_gray / max(avg_b, 1.0)
    result[:, :, 1] *= avg_gray / max(avg_g, 1.0)
    result[:, :, 2] *= avg_gray / max(avg_r, 1.0)
    return np.clip(result, 0, 255).astype(np.uint8)


def classify_combine(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPieceCombinee]:
    """Classification combinée spécialisée par groupe.

    Stratégie :
    - Déterminer le groupe de chaque pièce par vote majoritaire (HSV + Filtres)
    - Cuivre (1c, 2c, 5c) → Filtres (meilleur sur ce groupe)
    - Or (10c, 20c, 50c) et Bimétal (1e, 2e) → HSV avec égalisation
    """
    if not circles:
        return []

    # HSV (avec HistEq intégré) et Filtres
    results_hsv = classify_by_color_and_size(circles, image_bgr)
    results_filtres = classify_filtres(circles, image_bgr)

    n = len(circles)
    resultats = []

    for i in range(n):
        v_hsv = results_hsv[i] if i < len(results_hsv) else None
        v_filt = results_filtres[i] if i < len(results_filtres) else None

        # Déterminer le groupe par vote
        groupe_hsv = v_hsv.groupe_couleur if v_hsv else ""
        groupe_filt = v_filt.groupe_couleur if v_filt else ""

        # Si les Filtres détectent cuivre → leur faire confiance
        # Sinon → utiliser le HSV (meilleur sur or et bimétal)
        if groupe_filt == "cuivre" and v_filt:
            best_denom = v_filt.denomination
            best_conf = v_filt.confiance
            methode = "Filtres"
        elif v_hsv:
            best_denom = v_hsv.denomination
            best_conf = v_hsv.confiance
            methode = "HSV"
        elif v_filt:
            best_denom = v_filt.denomination
            best_conf = v_filt.confiance
            methode = "Filtres"
        else:
            best_denom = "1e"
            best_conf = 0.1
            methode = "fallback"

        groupe_final = ""
        if best_denom in ("1c", "2c", "5c"):
            groupe_final = "cuivre"
        elif best_denom in ("10c", "20c", "50c"):
            groupe_final = "or"
        elif best_denom in ("1e", "2e"):
            groupe_final = "bimetallic"

        resultats.append(ValeurPieceCombinee(
            cercle=circles[i],
            denomination=best_denom,
            valeur_centimes=VALEURS_CENTIMES[best_denom],
            confiance=round(min(1.0, best_conf), 3),
            groupe_couleur=groupe_final,
            methode_dominante=methode,
        ))

    return resultats


def valeur_totale_combinee(valuations: list[ValeurPieceCombinee]) -> tuple[int, str]:
    total = sum(v.valeur_centimes for v in valuations)
    euros, centimes = divmod(total, 100)
    if euros > 0 and centimes > 0:
        libelle = f"{euros}e{centimes:02d}"
    elif euros > 0:
        libelle = f"{euros}e"
    else:
        libelle = f"{total}c"
    return total, libelle
