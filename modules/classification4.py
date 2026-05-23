from __future__ import annotations

"""Classification combinée : ensemble de HSV + Filtres.

Combine les résultats de classification.py (HSV) et classification3.py (Filtres RGB)
avec sélection par confiance. Inclut une correction de balance des blancs (Gray-World).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from modules.segmentation import DetectedCircle
from modules.classification import classify_by_color_and_size, VALEURS_CENTIMES
from modules.classification3 import classify_filtres


@dataclass(frozen=True)
class ValeurPieceCombinee:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""
    methode_dominante: str = ""


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

    # Patch 2 : correction de balance des blancs avant classification
    image_wb = _correct_white_balance(image_bgr)

    # HSV (avec HistEq intégré) et Filtres
    results_hsv = classify_by_color_and_size(circles, image_wb)
    results_filtres = classify_filtres(circles, image_wb)

    n = len(circles)
    resultats = []

    for i in range(n):
        v_hsv = results_hsv[i] if i < len(results_hsv) else None
        v_filt = results_filtres[i] if i < len(results_filtres) else None

        # Patch 1 : sélection par confiance (plus de règle binaire groupe)
        # HSV est la méthode de base ; les Filtres peuvent prendre le dessus
        # seulement si leur confiance est significativement plus haute.
        groupe_hsv = v_hsv.groupe_couleur if v_hsv else ""
        groupe_filt = v_filt.groupe_couleur if v_filt else ""

        conf_hsv = v_hsv.confiance if v_hsv else 0.0
        conf_filt = v_filt.confiance if v_filt else 0.0

        # Les Filtres prennent le dessus si :
        #   - même groupe que HSV ET confiance Filtres > HSV + 0.08
        #   - OU groupe cuivre avec confiance Filtres nettement supérieure
        use_filtres = (
            v_filt is not None and v_hsv is not None
            and groupe_filt == groupe_hsv
            and conf_filt > conf_hsv + 0.08
        ) or (
            v_filt is not None
            and groupe_filt == "cuivre"
            and conf_filt > conf_hsv + 0.05
        )

        if use_filtres and v_filt:
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
