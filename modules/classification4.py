from __future__ import annotations

"""Classification combinée : ensemble de HSV + HLS + détection anneau Canny.

Combine les résultats de classification.py (HSV) et classification2.py (HLS)
avec un vote pondéré par la confiance, plus un détecteur d'anneau interne
basé sur Canny pour confirmer/infirmer le groupe bimétal.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from modules.segmentation import DetectedCircle, apply_clahe_bgr
from modules.classification import classify_by_color_and_size, ValeurPiece, VALEURS_CENTIMES
from modules.classification2 import classify_hls, ValeurPieceHLS


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


def classify_combine(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPieceCombinee]:
    """Classification par ensemble : HSV + HLS + détection anneau Canny."""
    if not circles:
        return []

    # Obtenir les résultats des deux méthodes
    results_hsv = classify_by_color_and_size(circles, image_bgr)
    results_hls = classify_hls(circles, image_bgr)

    n = len(circles)
    resultats = []

    for i in range(n):
        v_hsv = results_hsv[i] if i < len(results_hsv) else None
        v_hls = results_hls[i] if i < len(results_hls) else None

        # Détection de l'anneau interne (indicateur bimétal)
        score_anneau = _detecter_anneau_interne(image_bgr, circles[i])
        score_profil = _profil_radial_contraste(image_bgr, circles[i])
        score_bimetal_canny = 0.6 * score_anneau + 0.4 * score_profil

        # Collecter les votes
        votes: dict[str, float] = {}

        if v_hsv:
            poids_hsv = v_hsv.confiance * 1.0
            votes[v_hsv.denomination] = votes.get(v_hsv.denomination, 0.0) + poids_hsv

        if v_hls:
            poids_hls = v_hls.confiance * 0.9
            votes[v_hls.denomination] = votes.get(v_hls.denomination, 0.0) + poids_hls

        # Vote Canny : booste bimétal ou pénalise
        if score_bimetal_canny > 0.5:
            # Forte indication bimétal — booster 1e/2e
            # Choisir entre 1e et 2e selon le meilleur vote existant
            vote_1e = votes.get("1e", 0.0)
            vote_2e = votes.get("2e", 0.0)
            boost = score_bimetal_canny * 0.8
            if vote_2e >= vote_1e:
                votes["2e"] = votes.get("2e", 0.0) + boost
            else:
                votes["1e"] = votes.get("1e", 0.0) + boost
        elif score_bimetal_canny < 0.25:
            # Pas d'anneau → pénaliser bimétal
            penalite = (0.25 - score_bimetal_canny) * 0.6
            if "1e" in votes:
                votes["1e"] = max(0.01, votes["1e"] - penalite)
            if "2e" in votes:
                votes["2e"] = max(0.01, votes["2e"] - penalite)

        # Déterminer le gagnant
        if not votes:
            best_denom = "1e"
            best_conf = 0.1
        else:
            best_denom = max(votes, key=lambda d: votes[d])
            total_votes = sum(votes.values())
            best_conf = votes[best_denom] / total_votes if total_votes > 0 else 0.1

        # Déterminer la méthode dominante
        methode = "ensemble"
        if v_hsv and v_hls:
            if v_hsv.denomination == v_hls.denomination == best_denom:
                methode = "consensus"
            elif v_hsv and v_hsv.denomination == best_denom:
                methode = "HSV"
            elif v_hls and v_hls.denomination == best_denom:
                methode = "HLS"
            if score_bimetal_canny > 0.5 and best_denom in ("1e", "2e"):
                methode = "Canny+" + methode

        groupe = ""
        if best_denom in ("1c", "2c", "5c"):
            groupe = "cuivre"
        elif best_denom in ("10c", "20c", "50c"):
            groupe = "or"
        elif best_denom in ("1e", "2e"):
            groupe = "bimetallic"

        resultats.append(ValeurPieceCombinee(
            cercle=circles[i],
            denomination=best_denom,
            valeur_centimes=VALEURS_CENTIMES[best_denom],
            confiance=round(min(1.0, best_conf), 3),
            groupe_couleur=groupe,
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
