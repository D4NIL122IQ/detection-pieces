from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from modules.segmentation import DetectedCircle, apply_clahe_bgr


DIAMETRES_MM: dict[str, float] = {
    "1c":  16.25,
    "2c":  18.75,
    "5c":  21.25,
    "10c": 19.75,
    "20c": 22.25,
    "50c": 24.25,
    "1e":  23.25,
    "2e":  25.75,
}

VALEURS_CENTIMES: dict[str, int] = {
    "1c":    1,
    "2c":    2,
    "5c":    5,
    "10c":  10,
    "20c":  20,
    "50c":  50,
    "1e":  100,
    "2e":  200,
}

_GROUPES: dict[str, list[str]] = {
    "cuivre":     ["1c", "2c", "5c"],
    "or":         ["10c", "20c", "50c"],
    "bimetallic": ["1e", "2e"],
}

_DENOMINATIONS = list(DIAMETRES_MM.keys())
_DIAMS = np.array([DIAMETRES_MM[d] for d in _DENOMINATIONS], dtype=float)

_H_FRONTIERE = 17.0
_SIGMOID_PENTE = 0.55

# Paramètres de classification intra-groupe (couleur fine)
_INTRA_H_SIGMA = 4.0
_INTRA_S_SIGMA = 25.0
_FIABILITE_K   = 8.0

_CUIVRE_H_CENTERS: dict[str, float] = {"1c":  7.0, "2c": 11.0, "5c": 14.0}
_CUIVRE_S_CENTERS: dict[str, float] = {"1c": 90.0, "2c": 105.0, "5c": 120.0}
_OR_H_CENTERS:     dict[str, float] = {"10c": 20.0, "20c": 22.0, "50c": 24.0}
_OR_S_CENTERS:     dict[str, float] = {"10c": 100.0, "20c": 115.0, "50c": 130.0}

_BIMETALLIC_RATIO_PHYSIQUE = 0.108  # (2€ - 1€) / 1€ en diamètre

_HSV_OR     = {"h_min": 15, "h_max": 34, "s_min": 60, "v_min": 70}  # pour les pieces 1 et 2 € et 10 20 50 c
_HSV_ARGENT = {"s_max": 75, "v_min": 60} # pour les pieces 1 et 2 €
 

@dataclass(frozen=True)
class ValeurPiece:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""


def _pixels_hsv_normalises(
    image_bgr: np.ndarray,
    circle: DetectedCircle,
    ratio_ext: float = 0.88,
    ratio_int: float = 0.0,
) -> np.ndarray:
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

    hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
    return hsv[mask > 0]


def _stats_hsv(pixels: np.ndarray) -> dict:
    if len(pixels) == 0:
        return {"h": 0.0, "s": 0.0, "v_med": 0.0, "couv": 0.0, "fiable": False}

    h = pixels[:, 0].astype(float)
    s = pixels[:, 1].astype(float)
    v = pixels[:, 2].astype(float)

    v_med = float(np.median(v))
    v_bas  = max(20.0, v_med * 0.25)
    v_haut = min(250.0, v_med * 1.9)
    ok = (v >= v_bas) & (v <= v_haut)
    h_v, s_v = h[ok], s[ok]

    if len(h_v) == 0:
        return {"h": 0.0, "s": 0.0, "v_med": v_med, "couv": 0.0, "fiable": False}

    # Seuil adaptatif : en faible luminosité, abaisser le seuil de saturation
    seuil_sat = max(15.0, 35.0 * (v_med / 128.0))
    sature = s_v > seuil_sat
    couv = float(sature.sum()) / len(h_v)

    if sature.sum() >= 5:
        h_pond = float(np.average(h_v[sature], weights=s_v[sature]))
    else:
        h_pond = float(np.median(h_v))

    return {
        "h": h_pond,
        "s": float(np.mean(s_v)),
        "v_med": v_med,
        "couv": couv,
        "fiable": couv > 0.10 and len(h_v) > 8,
    }


def _score_groupe(stats: dict, groupe: str) -> float:
    h   = stats["h"]
    s   = stats["s"]
    cov = stats["couv"]
    v_med = stats.get("v_med", 128.0)

    if groupe == "cuivre":
        sig = 1.0 / (1.0 + np.exp(_SIGMOID_PENTE * (h - _H_FRONTIERE)))
        gau = float(np.exp(-0.5 * ((h - 12.0) / 5.0) ** 2))
        score_h = 0.55 * sig + 0.45 * gau
        score_s = float(np.clip((s - 45.0) / 65.0, 0.0, 1.0))
        return score_h * max(0.05, score_s)

    if groupe == "or":
        sig = 1.0 / (1.0 + np.exp(-_SIGMOID_PENTE * (h - _H_FRONTIERE)))
        gau = float(np.exp(-0.5 * ((h - 22.0) / 5.0) ** 2))
        score_h = 0.55 * sig + 0.45 * gau
        score_s = float(np.clip((s - 55.0) / 65.0, 0.0, 1.0))
        return score_h * max(0.05, score_s)

    # Bimétal : pénaliser en faible luminosité (la désaturation est un artefact)
    score_s   = float(np.clip(1.0 - s / 75.0, 0.0, 1.0))
    score_cov = float(np.clip(1.0 - cov / 0.28, 0.0, 1.0))
    score = 0.55 * score_s + 0.45 * score_cov

    # Pénalité : si V médian < 80, réduire le score bimétal
    if v_med < 100.0:
        penalite = float(np.clip(v_med / 100.0, 0.3, 1.0))
        score *= penalite

    return score


def _score_bimetallic(image_bgr: np.ndarray, circle: DetectedCircle) -> tuple[float, float]:
    px_c = _pixels_hsv_normalises(image_bgr, circle, ratio_ext=0.45)
    px_r = _pixels_hsv_normalises(image_bgr, circle, ratio_ext=0.88, ratio_int=0.52)

    def prop_or(px: np.ndarray) -> float:
        if len(px) == 0:
            return 0.0
        h, s, v = px[:, 0], px[:, 1], px[:, 2]
        v_med = max(30.0, float(np.median(v)) * 0.4)
        return float(np.mean(
            (h >= _HSV_OR["h_min"]) & (h <= _HSV_OR["h_max"]) &
            (s >= _HSV_OR["s_min"]) & (v >= v_med)
        ))

    def prop_argent(px: np.ndarray) -> float:
        if len(px) == 0:
            return 0.0
        s, v = px[:, 1], px[:, 2]
        v_med = max(30.0, float(np.median(v)) * 0.4)
        return float(np.mean((s <= _HSV_ARGENT["s_max"]) & (v >= v_med)))

    or_c, argent_c = prop_or(px_c), prop_argent(px_c)
    or_r, argent_r = prop_or(px_r), prop_argent(px_r)

    score_1e = argent_c + or_r
    score_2e = or_c     + argent_r
    total = score_1e + score_2e
    if total == 0:
        return 0.5, 0.5
    return score_1e / total, score_2e / total


def _meilleure_combinaison(radii: list[float], candidats: list[str]) -> tuple[list[str], float]:
    """Retourne (meilleure_combinaison, erreur_minimale).

    erreur_minimale == inf quand la taille ne peut pas être utilisée (n <= 1).
    """
    n = len(radii)
    m = len(candidats)

    if n == 0:
        return [], float("inf")

    if n > m:
        ref_r = radii[0]
        ref_d = DIAMETRES_MM[candidats[0]]
        result = []
        err = 0.0
        for idx, r in enumerate(radii):
            ratio_obs = r / ref_r
            best = min(candidats, key=lambda d: abs(DIAMETRES_MM[d] / ref_d - ratio_obs))
            result.append(best)
            ratio_exp = DIAMETRES_MM[best] / ref_d
            err += abs(ratio_obs - ratio_exp) / max(ratio_exp, 0.01)
        return result, err / n

    if n == m:
        diams = [DIAMETRES_MM[d] for d in candidats]
        err = 0.0
        pairs = 0
        for a in range(n):
            for b in range(a + 1, n):
                ratio_obs = radii[b] / radii[a]
                ratio_exp = diams[b] / diams[a]
                err += abs(ratio_obs - ratio_exp) / ratio_exp
                pairs += 1
        return list(candidats), err / max(pairs, 1)

    if n == 1:
        # Pas de comparaison possible — on retourne tous les candidats et err=inf
        return list(candidats), float("inf")

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


def _fiabilite_taille(radii: list[float], best_err: float) -> float:
    """Estime dans quelle mesure les ratios de taille sont fiables (0 = inutilisable, 1 = fiable)."""
    if len(radii) <= 1 or best_err == float("inf"):
        return 0.0

    arr = np.array(radii, dtype=float)
    mean_r = float(arr.mean())
    if mean_r == 0:
        return 0.0

    cv = float(arr.std()) / mean_r
    if cv < 0.04:
        # Toutes pièces de même taille → ratio inutile
        return 0.1

    return float(np.clip(np.exp(-_FIABILITE_K * best_err), 0.05, 1.0))


def _score_couleur_intragroupe(stats: dict, groupe: str) -> dict[str, float]:
    """Distribution de probabilité intra-groupe basée sur la couleur fine.

    Retourne une distribution normalisée sur les dénominations du groupe.
    Si les scores sont trop proches, retourne une distribution quasi-uniforme.
    """
    h = stats["h"]
    s = stats["s"]

    if groupe == "cuivre":
        denoms = ["1c", "2c", "5c"]
        scores = {}
        for d in denoms:
            hc = _CUIVRE_H_CENTERS[d]
            sc = _CUIVRE_S_CENTERS[d]
            score = (
                float(np.exp(-0.5 * ((h - hc) / _INTRA_H_SIGMA) ** 2)) *
                float(np.exp(-0.5 * ((s - sc) / _INTRA_S_SIGMA) ** 2))
            )
            scores[d] = max(score, 1e-6)

    elif groupe == "or":
        denoms = ["10c", "20c", "50c"]
        scores = {}
        for d in denoms:
            hc = _OR_H_CENTERS[d]
            sc = _OR_S_CENTERS[d]
            score = (
                float(np.exp(-0.5 * ((h - hc) / _INTRA_H_SIGMA) ** 2)) *
                float(np.exp(-0.5 * ((s - sc) / _INTRA_S_SIGMA) ** 2))
            )
            scores[d] = max(score, 1e-6)

    else:
        return {}

    total = sum(scores.values())
    return {d: v / total for d, v in scores.items()}


def classify_by_relative_size(circles: list[DetectedCircle]) -> list[ValeurPiece]:
    if not circles:
        return []

    radii = np.array([c.radius for c in circles], dtype=float)
    radii_norm = radii / radii.min()
    min_d = min(DIAMETRES_MM.values())
    ratios = np.array([DIAMETRES_MM[d] / min_d for d in _DENOMINATIONS])
    tol = 0.15

    resultats = []
    for circle, r_norm in zip(circles, radii_norm):
        distances = np.abs(ratios - r_norm)
        idx = int(np.argmin(distances))
        denom = _DENOMINATIONS[idx]
        conf = max(0.0, 1.0 - float(distances[idx]) / tol)
        resultats.append(ValeurPiece(
            cercle=circle, denomination=denom,
            valeur_centimes=VALEURS_CENTIMES[denom], confiance=round(conf, 3),
        ))
    return resultats


def classify_with_reference(
    circles: list[DetectedCircle],
    cercle_reference: DetectedCircle,
    denomination_reference: str,
) -> list[ValeurPiece]:
    if not circles:
        return []
    if denomination_reference not in DIAMETRES_MM:
        raise KeyError(f"Dénomination inconnue : {denomination_reference!r}.")
    if cercle_reference.radius <= 0:
        raise ValueError("Rayon de référence nul.")

    mm_px = DIAMETRES_MM[denomination_reference] / (2 * cercle_reference.radius)
    tol_mm = 1.5

    resultats = []
    for circle in circles:
        diam = 2 * circle.radius * mm_px
        distances = np.abs(_DIAMS - diam)
        idx = int(np.argmin(distances))
        denom = _DENOMINATIONS[idx]
        conf = max(0.0, 1.0 - float(distances[idx]) / tol_mm)
        resultats.append(ValeurPiece(
            cercle=circle, denomination=denom,
            valeur_centimes=VALEURS_CENTIMES[denom], confiance=round(conf, 3),
        ))
    return resultats


def classify_by_color_and_size(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPiece]:
    if not circles:
        return []

    n = len(circles)

    all_stats = [
        _stats_hsv(_pixels_hsv_normalises(image_bgr, c, ratio_ext=0.85))
        for c in circles
    ]

    groupes_ordre = list(_GROUPES.keys())
    groupe_scores = np.array([
        [_score_groupe(st, g) for g in groupes_ordre]
        for st in all_stats
    ])

    totaux = groupe_scores.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1.0
    groupe_proba = groupe_scores / totaux

    assigned_groupes = [groupes_ordre[int(np.argmax(groupe_proba[i]))] for i in range(n)]

    bi_scores = [_score_bimetallic(image_bgr, c) for c in circles]

    final_denoms: list[str | None] = [None] * n
    final_confs:  list[float]       = [0.0]  * n

    for groupe in groupes_ordre:
        candidats = _GROUPES[groupe]
        indices = [i for i in range(n) if assigned_groupes[i] == groupe]

        if not indices:
            continue

        conf_groupe = float(np.mean([groupe_proba[i, groupes_ordre.index(groupe)] for i in indices]))

        if groupe == "bimetallic":
            radii_bi = [float(circles[i].radius) for i in indices]
            if len(indices) >= 2:
                r_max = max(radii_bi)
                r_min = min(radii_bi)
                ratio_spread = (r_max - r_min) / r_min if r_min > 0 else 0.0
                if ratio_spread < 0.25:
                    fiabilite_bi = min(1.0, ratio_spread / _BIMETALLIC_RATIO_PHYSIQUE)
                else:
                    fiabilite_bi = 0.3
                size_boost = fiabilite_bi * 0.25
            else:
                size_boost = 0.0

            for i in indices:
                s1e, s2e = bi_scores[i]
                if size_boost > 0:
                    autres = [circles[j].radius for j in indices if j != i]
                    r_i = circles[i].radius
                    if all(r_i >= r_j for r_j in autres):
                        s2e = min(1.0, s2e + size_boost)
                    else:
                        s1e = min(1.0, s1e + size_boost)

                denom = "2e" if s2e > s1e else "1e"
                final_denoms[i] = denom
                final_confs[i]  = round(conf_groupe * max(s1e, s2e), 3)

        else:
            indices_tries = sorted(indices, key=lambda i: circles[i].radius)
            radii_tries   = [float(circles[i].radius) for i in indices_tries]

            combo, best_err = _meilleure_combinaison(radii_tries, candidats)
            fiabilite = _fiabilite_taille(radii_tries, best_err)

            for pos, i in enumerate(indices_tries):
                score_intra = _score_couleur_intragroupe(all_stats[i], groupe)

                if not score_intra:
                    # Groupe non supporté — fallback
                    final_denoms[i] = combo[pos] if pos < len(combo) else candidats[len(candidats) // 2]
                    final_confs[i]  = round(conf_groupe * (0.4 + 0.4 * fiabilite), 3)
                    continue

                # Boost multiplicatif de la dénomination suggérée par la taille
                scores = dict(score_intra)
                if fiabilite > 0 and pos < len(combo):
                    denom_taille = combo[pos]
                    scores[denom_taille] = scores.get(denom_taille, 1e-6) * (1.0 + fiabilite * 2.0)
                    total = sum(scores.values())
                    scores = {d: v / total for d, v in scores.items()}

                best_denom = max(scores, key=lambda d: scores[d])
                best_score = scores[best_denom]
                autres_scores = [v for d, v in scores.items() if d != best_denom]
                marge = best_score - (sum(autres_scores) / max(len(autres_scores), 1))

                final_denoms[i] = best_denom
                final_confs[i]  = round(
                    min(1.0, conf_groupe * (0.4 + 0.4 * fiabilite) * (1.0 + 0.2 * max(0.0, marge))),
                    3,
                )

    for i in range(n):
        if final_denoms[i] is None:
            final_denoms[i] = _DENOMINATIONS[int(np.argmax(groupe_scores[i]))]
            final_confs[i]  = 0.1

    resultats = []
    for i, circle in enumerate(circles):
        denom = final_denoms[i]
        groupe = next(
            (g for g, ds in _GROUPES.items() if denom in ds),
            "",
        )
        resultats.append(ValeurPiece(
            cercle=circle,
            denomination=denom,
            valeur_centimes=VALEURS_CENTIMES[denom],
            confiance=min(1.0, final_confs[i]),
            groupe_couleur=groupe,
        ))

    return resultats


def valeur_totale(valuations: list[ValeurPiece]) -> tuple[int, str]:
    total = sum(v.valeur_centimes for v in valuations)
    euros, centimes = divmod(total, 100)
    if euros > 0 and centimes > 0:
        libelle = f"{euros}e{centimes:02d}"
    elif euros > 0:
        libelle = f"{euros}e"
    else:
        libelle = f"{total}c"
    return total, libelle
