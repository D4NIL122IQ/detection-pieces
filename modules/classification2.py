from __future__ import annotations

"""Classification des pièces d'euros par analyse HLS (Hue, Lightness, Saturation).

Approche alternative à classification.py (HSV). Utilise l'espace colorimétrique HLS
pour déterminer le groupe (cuivre/or/bimétal) puis la valeur exacte.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from modules.constants import DIAMETRES_MM, VALEURS_CENTIMES, GROUPES as _GROUPES
from modules.segmentation import DetectedCircle, apply_clahe_bgr

# --- Seuils HLS pour chaque groupe ---
# OpenCV HLS : H [0,180], L [0,255], S [0,255]
# Cuivre : teinte rouge-orangé (H ~ 5-15 en OpenCV HLS), saturation élevée
# Or nordique : teinte jaune-doré (H ~ 15-30), saturation élevée
# Bimétal : saturation faible (aspect gris/argenté dominant)

_H_CUIVRE_CENTER = 10.0
_H_CUIVRE_SIGMA = 6.0
_H_OR_CENTER = 22.0
_H_OR_SIGMA = 6.0
_H_FRONTIERE = 15.0
_SIGMOID_PENTE = 0.5

# Seuils intra-groupe pour différencier les pièces de même classe
# Cuivre : 1c plus rouge/sombre, 5c plus orangé/clair
_CUIVRE_H_CENTERS: dict[str, float] = {"1c": 6.0, "2c": 10.0, "5c": 14.0}
_CUIVRE_L_CENTERS: dict[str, float] = {"1c": 95.0, "2c": 110.0, "5c": 125.0}
_CUIVRE_S_CENTERS: dict[str, float] = {"1c": 80.0, "2c": 100.0, "5c": 115.0}

# Or : 10c plus petit et légèrement plus pâle, 50c plus saturé
_OR_H_CENTERS: dict[str, float] = {"10c": 18.0, "20c": 22.0, "50c": 25.0}
_OR_L_CENTERS: dict[str, float] = {"10c": 120.0, "20c": 130.0, "50c": 140.0}
_OR_S_CENTERS: dict[str, float] = {"10c": 95.0, "20c": 110.0, "50c": 125.0}

_INTRA_H_SIGMA = 4.5
_INTRA_L_SIGMA = 20.0
_INTRA_S_SIGMA = 25.0


@dataclass(frozen=True)
class ValeurPieceHLS:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""


def _pixels_hls(
    image_bgr: np.ndarray,
    circle: DetectedCircle,
    ratio_ext: float = 0.85,
    ratio_int: float = 0.0,
) -> np.ndarray:
    """Extrait les pixels HLS dans la région circulaire de la pièce."""
    cx, cy, r = circle.x, circle.y, circle.radius
    r_ext = max(1, int(r * ratio_ext))

    x1 = max(0, cx - r_ext)
    y1 = max(0, cy - r_ext)
    x2 = min(image_bgr.shape[1], cx + r_ext)
    y2 = min(image_bgr.shape[0], cy + r_ext)

    roi = image_bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    roi_norm = apply_clahe_bgr(roi, clip_limit=3.0, tile_grid_size=(4, 4))

    lx, ly = cx - x1, cy - y1
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), r_ext, 255, -1)
    if ratio_int > 0:
        r_int = max(1, int(r * ratio_int))
        cv2.circle(mask, (lx, ly), r_int, 0, -1)

    hls = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HLS)
    return hls[mask > 0]


def _stats_hls(pixels: np.ndarray) -> dict:
    """Calcule les statistiques HLS des pixels extraits."""
    if len(pixels) == 0:
        return {"h": 0.0, "l": 0.0, "s": 0.0, "couv": 0.0, "fiable": False}

    h = pixels[:, 0].astype(float)  # Hue [0,180]
    l = pixels[:, 1].astype(float)  # Lightness [0,255]
    s = pixels[:, 2].astype(float)  # Saturation [0,255]

    # Filtrer pixels trop sombres ou trop clairs (non informatifs)
    l_med = float(np.median(l))
    l_bas = max(30.0, l_med * 0.3)
    l_haut = min(240.0, l_med * 1.8)
    ok = (l >= l_bas) & (l <= l_haut)
    h_v, l_v, s_v = h[ok], l[ok], s[ok]

    if len(h_v) == 0:
        return {"h": 0.0, "l": 0.0, "s": 0.0, "couv": 0.0, "fiable": False}

    l_med = float(np.median(l_v))

    # Seuil adaptatif : en faible luminosité, abaisser le seuil de saturation
    seuil_sat = max(18.0, 40.0 * (l_med / 128.0))
    sature = s_v > seuil_sat
    couv = float(sature.sum()) / len(h_v)

    if sature.sum() >= 5:
        h_pond = float(np.average(h_v[sature], weights=s_v[sature]))
        s_moy = float(np.mean(s_v[sature]))
    else:
        h_pond = float(np.median(h_v))
        s_moy = float(np.mean(s_v))

    l_moy = float(np.mean(l_v))

    return {
        "h": h_pond,
        "l": l_moy,
        "l_med": l_med,
        "s": s_moy,
        "couv": couv,
        "fiable": couv > 0.10 and len(h_v) > 8,
    }


def _score_groupe_hls(stats: dict, groupe: str) -> float:
    """Score d'appartenance à un groupe basé sur HLS."""
    h = stats["h"]
    s = stats["s"]
    couv = stats["couv"]
    l_med = stats.get("l_med", 128.0)

    if groupe == "cuivre":
        # Sigmoïde : favorise H < frontière
        sig = 1.0 / (1.0 + np.exp(_SIGMOID_PENTE * (h - _H_FRONTIERE)))
        # Gaussienne centrée sur la teinte cuivre
        gau = float(np.exp(-0.5 * ((h - _H_CUIVRE_CENTER) / _H_CUIVRE_SIGMA) ** 2))
        score_h = 0.5 * sig + 0.5 * gau
        # La saturation doit être élevée pour cuivre
        score_s = float(np.clip((s - 40.0) / 80.0, 0.0, 1.0))
        return score_h * max(0.05, score_s)

    if groupe == "or":
        sig = 1.0 / (1.0 + np.exp(-_SIGMOID_PENTE * (h - _H_FRONTIERE)))
        gau = float(np.exp(-0.5 * ((h - _H_OR_CENTER) / _H_OR_SIGMA) ** 2))
        score_h = 0.5 * sig + 0.5 * gau
        score_s = float(np.clip((s - 50.0) / 80.0, 0.0, 1.0))
        return score_h * max(0.05, score_s)

    # Bimétal : faible saturation + faible couverture de pixels saturés
    score_s = float(np.clip(1.0 - s / 80.0, 0.0, 1.0))
    score_cov = float(np.clip(1.0 - couv / 0.30, 0.0, 1.0))
    score = 0.55 * score_s + 0.45 * score_cov

    # Pénalité en faible luminosité (la désaturation est un artefact, pas du bimétal)
    if l_med < 100.0:
        penalite = float(np.clip(l_med / 100.0, 0.3, 1.0))
        score *= penalite

    return score


def _score_bimetallic_hls(image_bgr: np.ndarray, circle: DetectedCircle) -> tuple[float, float]:
    """Distingue 1€ vs 2€ par analyse centre/couronne en HLS."""
    px_centre = _pixels_hls(image_bgr, circle, ratio_ext=0.45)
    px_couronne = _pixels_hls(image_bgr, circle, ratio_ext=0.88, ratio_int=0.52)

    def prop_dore(px: np.ndarray) -> float:
        if len(px) == 0:
            return 0.0
        h, l, s = px[:, 0], px[:, 1], px[:, 2]
        l_med = max(30.0, float(np.median(l)) * 0.4)
        return float(np.mean(
            (h >= 12) & (h <= 32) & (s >= 55) & (l >= l_med)
        ))

    def prop_argent(px: np.ndarray) -> float:
        if len(px) == 0:
            return 0.0
        l, s = px[:, 1], px[:, 2]
        l_med = max(30.0, float(np.median(l)) * 0.4)
        return float(np.mean((s <= 70) & (l >= l_med)))

    or_c, argent_c = prop_dore(px_centre), prop_argent(px_centre)
    or_r, argent_r = prop_dore(px_couronne), prop_argent(px_couronne)

    # 1€ : centre argent + couronne or
    # 2€ : centre or + couronne argent
    score_1e = argent_c + or_r
    score_2e = or_c + argent_r
    total = score_1e + score_2e
    if total == 0:
        return 0.5, 0.5
    return score_1e / total, score_2e / total


def _score_intragroupe_hls(stats: dict, groupe: str) -> dict[str, float]:
    """Score intra-groupe basé sur H, L et S pour différencier les pièces de même classe."""
    h = stats["h"]
    l = stats["l"]
    s = stats["s"]

    if groupe == "cuivre":
        denoms = ["1c", "2c", "5c"]
        scores = {}
        for d in denoms:
            score = (
                float(np.exp(-0.5 * ((h - _CUIVRE_H_CENTERS[d]) / _INTRA_H_SIGMA) ** 2)) *
                float(np.exp(-0.5 * ((l - _CUIVRE_L_CENTERS[d]) / _INTRA_L_SIGMA) ** 2)) *
                float(np.exp(-0.5 * ((s - _CUIVRE_S_CENTERS[d]) / _INTRA_S_SIGMA) ** 2))
            )
            scores[d] = max(score, 1e-6)
    elif groupe == "or":
        denoms = ["10c", "20c", "50c"]
        scores = {}
        for d in denoms:
            score = (
                float(np.exp(-0.5 * ((h - _OR_H_CENTERS[d]) / _INTRA_H_SIGMA) ** 2)) *
                float(np.exp(-0.5 * ((l - _OR_L_CENTERS[d]) / _INTRA_L_SIGMA) ** 2)) *
                float(np.exp(-0.5 * ((s - _OR_S_CENTERS[d]) / _INTRA_S_SIGMA) ** 2))
            )
            scores[d] = max(score, 1e-6)
    else:
        return {}

    total = sum(scores.values())
    return {d: v / total for d, v in scores.items()}


def _meilleure_combinaison_taille(radii: list[float], candidats: list[str]) -> tuple[list[str], float]:
    """Trouve la meilleure assignation par ratios de taille."""
    from itertools import combinations

    n = len(radii)
    m = len(candidats)

    if n == 0:
        return [], float("inf")
    if n == 1:
        return list(candidats), float("inf")

    if n >= m:
        diams = [DIAMETRES_MM[d] for d in candidats]
        ref_r = radii[0]
        ref_d = diams[0]
        result = []
        err = 0.0
        for r in radii:
            ratio_obs = r / ref_r
            best = min(candidats, key=lambda d: abs(DIAMETRES_MM[d] / ref_d - ratio_obs))
            result.append(best)
            ratio_exp = DIAMETRES_MM[best] / ref_d
            err += abs(ratio_obs - ratio_exp) / max(ratio_exp, 0.01)
        return result, err / n

    best_combo: list[str] = list(candidats[:n])
    best_err = float("inf")

    for combo in combinations(candidats, n):
        diams = [DIAMETRES_MM[d] for d in combo]
        err = 0.0
        pairs = 0
        for a in range(n):
            for b in range(a + 1, n):
                ratio_obs = radii[b] / radii[a]
                ratio_exp = diams[b] / diams[a]
                err += abs(ratio_obs - ratio_exp) / ratio_exp
                pairs += 1
        err /= max(pairs, 1)
        if err < best_err:
            best_err = err
            best_combo = list(combo)

    return best_combo, best_err


def classify_hls(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPieceHLS]:
    """Classification principale par HLS : groupe couleur + taille relative."""
    if not circles:
        return []

    n = len(circles)

    # Calcul des stats HLS pour chaque pièce
    all_stats = [_stats_hls(_pixels_hls(image_bgr, c)) for c in circles]

    # Scores de groupe
    groupes_ordre = list(_GROUPES.keys())
    groupe_scores = np.array([
        [_score_groupe_hls(st, g) for g in groupes_ordre]
        for st in all_stats
    ])

    totaux = groupe_scores.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1.0
    groupe_proba = groupe_scores / totaux

    assigned_groupes = [groupes_ordre[int(np.argmax(groupe_proba[i]))] for i in range(n)]

    # Classification par groupe
    final_denoms: list[str | None] = [None] * n
    final_confs: list[float] = [0.0] * n

    for groupe in groupes_ordre:
        candidats = _GROUPES[groupe]
        indices = [i for i in range(n) if assigned_groupes[i] == groupe]
        if not indices:
            continue

        conf_groupe = float(np.mean([groupe_proba[i, groupes_ordre.index(groupe)] for i in indices]))

        if groupe == "bimetallic":
            for i in indices:
                s1e, s2e = _score_bimetallic_hls(image_bgr, circles[i])
                # Bonus taille si plusieurs bimétaux
                if len(indices) >= 2:
                    autres_r = [circles[j].radius for j in indices if j != i]
                    if all(circles[i].radius >= r for r in autres_r):
                        s2e = min(1.0, s2e + 0.15)
                    elif all(circles[i].radius <= r for r in autres_r):
                        s1e = min(1.0, s1e + 0.15)

                denom = "2e" if s2e > s1e else "1e"
                final_denoms[i] = denom
                final_confs[i] = round(conf_groupe * max(s1e, s2e), 3)
        else:
            indices_tries = sorted(indices, key=lambda i: circles[i].radius)
            radii_tries = [float(circles[i].radius) for i in indices_tries]

            combo, best_err = _meilleure_combinaison_taille(radii_tries, candidats)
            fiabilite = 0.0
            if len(radii_tries) > 1 and best_err != float("inf"):
                fiabilite = float(np.clip(np.exp(-8.0 * best_err), 0.05, 1.0))

            for pos, i in enumerate(indices_tries):
                score_intra = _score_intragroupe_hls(all_stats[i], groupe)

                if not score_intra:
                    final_denoms[i] = combo[pos] if pos < len(combo) else candidats[len(candidats) // 2]
                    final_confs[i] = round(conf_groupe * 0.5, 3)
                    continue

                # Combiner couleur intra-groupe + taille
                scores = dict(score_intra)
                if fiabilite > 0 and pos < len(combo):
                    denom_taille = combo[pos]
                    scores[denom_taille] = scores.get(denom_taille, 1e-6) * (1.0 + fiabilite * 2.0)
                    total = sum(scores.values())
                    scores = {d: v / total for d, v in scores.items()}

                best_denom = max(scores, key=lambda d: scores[d])
                final_denoms[i] = best_denom
                final_confs[i] = round(conf_groupe * (0.4 + 0.4 * fiabilite), 3)

    # Fallback
    for i in range(n):
        if final_denoms[i] is None:
            final_denoms[i] = "1e"
            final_confs[i] = 0.1

    resultats = []
    for i, circle in enumerate(circles):
        denom = final_denoms[i]
        groupe = next((g for g, ds in _GROUPES.items() if denom in ds), "")
        resultats.append(ValeurPieceHLS(
            cercle=circle,
            denomination=denom,
            valeur_centimes=VALEURS_CENTIMES[denom],
            confiance=min(1.0, final_confs[i]),
            groupe_couleur=groupe,
        ))

    return resultats


from modules.constants import valeur_totale as valeur_totale_hls  # noqa: E402, F811
