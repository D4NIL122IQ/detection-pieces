from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from modules.segmentation import DetectedCircle, apply_clahe_bgr


DIAMETRES_MM: dict[str, float] = {
    "1c": 16.25,
    "2c": 18.75,
    "5c": 21.25,
    "10c": 19.75,
    "20c": 22.25,
    "50c": 24.25,
    "1e": 23.25,
    "2e": 25.75,
}

VALEURS_CENTIMES: dict[str, int] = {
    "1c": 1,
    "2c": 2,
    "5c": 5,
    "10c": 10,
    "20c": 20,
    "50c": 50,
    "1e": 100,
    "2e": 200,
}

_GROUPES: dict[str, list[str]] = {
    "cuivre": ["1c", "2c", "5c"],
    "or": ["10c", "20c", "50c"],
    "bimetallic": ["1e", "2e"],
}

_DENOMINATIONS = list(DIAMETRES_MM.keys())
_DIAMS = np.array([DIAMETRES_MM[d] for d in _DENOMINATIONS], dtype=float)

_INTRA_H_SIGMA = 4.5
_INTRA_S_SIGMA = 24.0
_FIABILITE_K = 8.0

_CUIVRE_H_CENTERS: dict[str, float] = {"1c": 7.0, "2c": 10.5, "5c": 14.0}
_CUIVRE_S_CENTERS: dict[str, float] = {"1c": 75.0, "2c": 92.0, "5c": 108.0}
_OR_H_CENTERS: dict[str, float] = {"10c": 18.0, "20c": 22.0, "50c": 26.0}
_OR_S_CENTERS: dict[str, float] = {"10c": 68.0, "20c": 84.0, "50c": 104.0}


@dataclass(frozen=True)
class ValeurPiece:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""


def _weighted_circular_mean(h: np.ndarray, weights: np.ndarray) -> float:
    if len(h) == 0:
        return 0.0
    angles = h.astype(float) * (2.0 * np.pi / 180.0)
    w = np.maximum(weights.astype(float), 1e-6)
    x = float(np.sum(np.cos(angles) * w))
    y = float(np.sum(np.sin(angles) * w))
    if x == 0.0 and y == 0.0:
        return float(np.median(h))
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * (180.0 / (2.0 * np.pi)))


def _gauss(x: float, mean: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mean) / sigma) ** 2))


def _extract_coin_samples(image_bgr: np.ndarray, circle: DetectedCircle, ratio_ext: float = 0.88) -> dict:
    cx, cy, r = circle.x, circle.y, circle.radius
    r_ext = max(1, int(r * ratio_ext))

    x1 = max(0, cx - r_ext)
    y1 = max(0, cy - r_ext)
    x2 = min(image_bgr.shape[1], cx + r_ext)
    y2 = min(image_bgr.shape[0], cy + r_ext)

    roi = image_bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return {
            "hsv": np.empty((0, 3), dtype=np.uint8),
            "lab_ab": np.empty((0, 2), dtype=np.float32),
            "d_norm": np.empty((0,), dtype=np.float32),
        }

    roi_norm = apply_clahe_bgr(roi, clip_limit=3.0, tile_grid_size=(4, 4))

    yy, xx = np.indices(roi_norm.shape[:2])
    lx = cx - x1
    ly = cy - y1
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    mask = dist <= r_ext
    d_norm = (dist[mask] / max(r_ext, 1)).astype(np.float32)

    hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)[mask]
    lab = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2LAB)[mask]
    lab_ab = lab[:, 1:3].astype(np.float32)

    if len(hsv) == 0:
        return {
            "hsv": np.empty((0, 3), dtype=np.uint8),
            "lab_ab": np.empty((0, 2), dtype=np.float32),
            "d_norm": np.empty((0,), dtype=np.float32),
        }

    v = hsv[:, 2].astype(float)
    v_low = np.percentile(v, 8)
    v_high = np.percentile(v, 96)
    valid = (v >= v_low) & (v <= v_high)

    return {
        "hsv": hsv[valid],
        "lab_ab": lab_ab[valid],
        "d_norm": d_norm[valid],
    }


def _cluster_type(h_mean: float, s_mean: float, a_mean: float, b_mean: float) -> str:
    a_dev = a_mean - 128.0
    b_dev = b_mean - 128.0
    chroma = float(np.hypot(a_dev, b_dev))

    if s_mean < 42.0 or chroma < 10.0:
        return "neutral"

    goldness = b_dev - 0.65 * a_dev
    copperness = a_dev - 0.45 * b_dev

    if goldness > copperness + 1.5:
        return "gold"
    if copperness > goldness + 1.5:
        return "copper"

    if h_mean >= 16.0 and b_dev >= a_dev:
        return "gold"
    return "copper"


def _build_descriptor(image_bgr: np.ndarray, circle: DetectedCircle) -> dict:
    samples = _extract_coin_samples(image_bgr, circle)
    hsv = samples["hsv"]
    lab_ab = samples["lab_ab"]
    d_norm = samples["d_norm"]

    if len(hsv) == 0:
        return {
            "group": "or",
            "group_conf": 0.0,
            "score_1e": 0.5,
            "score_2e": 0.5,
            "h_dom": 20.0,
            "s_dom": 60.0,
            "type_props": {"copper": 0.0, "gold": 0.0, "neutral": 1.0},
            "zone_props": {
                "center": {"copper": 0.0, "gold": 0.0, "neutral": 1.0},
                "ring": {"copper": 0.0, "gold": 0.0, "neutral": 1.0},
            },
        }

    n = len(hsv)
    k = min(3, n)
    if k <= 1:
        labels = np.zeros((n,), dtype=np.int32)
        centers = np.mean(lab_ab, axis=0, keepdims=True)
    else:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            25,
            0.25,
        )
        _compactness, labels_raw, centers = cv2.kmeans(
            lab_ab,
            k,
            None,
            criteria,
            6,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels_raw.reshape(-1).astype(np.int32)

    types: list[str] = []
    weights: list[float] = []
    cluster_h: list[float] = []
    cluster_s: list[float] = []

    for idx in range(len(centers)):
        sel = labels == idx
        if not np.any(sel):
            types.append("neutral")
            weights.append(0.0)
            cluster_h.append(0.0)
            cluster_s.append(0.0)
            continue
        px = hsv[sel]
        h_mean = _weighted_circular_mean(px[:, 0], np.maximum(px[:, 1].astype(float), 1.0))
        s_mean = float(np.median(px[:, 1]))
        a_mean = float(centers[idx][0])
        b_mean = float(centers[idx][1])
        types.append(_cluster_type(h_mean, s_mean, a_mean, b_mean))
        weights.append(float(np.mean(sel)))
        cluster_h.append(h_mean)
        cluster_s.append(s_mean)

    def _props(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {"copper": 0.0, "gold": 0.0, "neutral": 0.0}
        local = labels[mask]
        total = float(len(local))
        out = {"copper": 0.0, "gold": 0.0, "neutral": 0.0}
        for idx, typ in enumerate(types):
            out[typ] += float(np.sum(local == idx)) / total
        return out

    center_mask = d_norm <= 0.45
    ring_mask = (d_norm >= 0.52) & (d_norm <= 0.88)
    all_mask = np.ones((len(d_norm),), dtype=bool)

    props_all = _props(all_mask)
    props_center = _props(center_mask)
    props_ring = _props(ring_mask)

    color_clusters = [i for i, t in enumerate(types) if t in ("copper", "gold") and weights[i] > 0]
    if color_clusters:
        best_idx = max(color_clusters, key=lambda i: weights[i] * max(cluster_s[i], 1.0))
        h_dom = cluster_h[best_idx]
        s_dom = cluster_s[best_idx]
    else:
        h_dom = _weighted_circular_mean(hsv[:, 0], np.maximum(hsv[:, 1].astype(float), 1.0))
        s_dom = float(np.median(hsv[:, 1]))

    score_1e = 0.65 * props_center["neutral"] + 0.35 * props_ring["gold"]
    score_2e = 0.65 * props_center["gold"] + 0.35 * props_ring["neutral"]
    contrast_bi = (
        abs(props_center["gold"] - props_ring["gold"]) +
        abs(props_center["neutral"] - props_ring["neutral"])
    ) / 2.0
    bimetal_score = max(score_1e, score_2e) * 0.7 + contrast_bi * 0.3

    if (
        bimetal_score > 0.56 and
        props_all["gold"] > 0.16 and
        props_all["neutral"] > 0.16
    ):
        group = "bimetallic"
        group_conf = min(1.0, bimetal_score)
    else:
        if props_all["copper"] > props_all["gold"] + 0.14:
            group = "cuivre"
            group_conf = props_all["copper"]
        elif props_all["gold"] > props_all["copper"] + 0.02:
            group = "or"
            group_conf = props_all["gold"]
        else:
            group = "or" if h_dom >= 14.0 else "cuivre"
            group_conf = max(props_all["copper"], props_all["gold"], 0.25)

    return {
        "group": group,
        "group_conf": float(group_conf),
        "score_1e": float(score_1e),
        "score_2e": float(score_2e),
        "h_dom": float(h_dom),
        "s_dom": float(s_dom),
        "type_props": props_all,
        "zone_props": {"center": props_center, "ring": props_ring},
    }


def _meilleure_combinaison(radii: list[float], candidats: list[str]) -> tuple[list[str], float]:
    n = len(radii)
    m = len(candidats)

    if n == 0:
        return [], float("inf")

    if n > m:
        ref_r = radii[0]
        ref_d = DIAMETRES_MM[candidats[0]]
        result = []
        err = 0.0
        for r in radii:
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
        return list(candidats), float("inf")

    best_combo = list(candidats[:n])
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
    if len(radii) <= 1 or best_err == float("inf"):
        return 0.0

    arr = np.array(radii, dtype=float)
    mean_r = float(arr.mean())
    if mean_r == 0.0:
        return 0.0

    cv = float(arr.std()) / mean_r
    if cv < 0.04:
        return 0.1

    return float(np.clip(np.exp(-_FIABILITE_K * best_err), 0.05, 1.0))


def _score_couleur_intragroupe(desc: dict, groupe: str) -> dict[str, float]:
    h = desc["h_dom"]
    s = desc["s_dom"]

    if groupe == "cuivre":
        denoms = ["1c", "2c", "5c"]
        scores = {}
        for d in denoms:
            score = _gauss(h, _CUIVRE_H_CENTERS[d], _INTRA_H_SIGMA) * _gauss(s, _CUIVRE_S_CENTERS[d], _INTRA_S_SIGMA)
            scores[d] = max(score, 1e-6)
    elif groupe == "or":
        denoms = ["10c", "20c", "50c"]
        scores = {}
        for d in denoms:
            score = _gauss(h, _OR_H_CENTERS[d], _INTRA_H_SIGMA) * _gauss(s, _OR_S_CENTERS[d], _INTRA_S_SIGMA)
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
        resultats.append(
            ValeurPiece(
                cercle=circle,
                denomination=denom,
                valeur_centimes=VALEURS_CENTIMES[denom],
                confiance=round(conf, 3),
            )
        )
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
        resultats.append(
            ValeurPiece(
                cercle=circle,
                denomination=denom,
                valeur_centimes=VALEURS_CENTIMES[denom],
                confiance=round(conf, 3),
            )
        )
    return resultats


def classify_by_color_and_size(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPiece]:
    if not circles:
        return []

    descriptors = [_build_descriptor(image_bgr, c) for c in circles]

    group_indices: dict[str, list[int]] = {"cuivre": [], "or": [], "bimetallic": []}
    for i, desc in enumerate(descriptors):
        group_indices[desc["group"]].append(i)

    final_denoms: list[str | None] = [None] * len(circles)
    final_confs: list[float] = [0.0] * len(circles)

    for groupe, candidats in _GROUPES.items():
        indices = group_indices[groupe]
        if not indices:
            continue

        if groupe == "bimetallic":
            radii = [float(circles[i].radius) for i in indices]
            r_min = min(radii) if radii else 0.0
            r_max = max(radii) if radii else 0.0
            spread = ((r_max - r_min) / r_min) if r_min > 0 else 0.0
            size_boost = min(0.22, spread * 0.8)

            for i in indices:
                desc = descriptors[i]
                s1e = desc["score_1e"]
                s2e = desc["score_2e"]
                if len(indices) >= 2 and size_boost > 0.0:
                    if circles[i].radius >= np.median(radii):
                        s2e += size_boost
                    else:
                        s1e += size_boost
                denom = "2e" if s2e > s1e else "1e"
                conf = min(1.0, desc["group_conf"] * max(s1e, s2e))
                final_denoms[i] = denom
                final_confs[i] = round(conf, 3)
            continue

        indices_sorted = sorted(indices, key=lambda i: circles[i].radius)
        radii_sorted = [float(circles[i].radius) for i in indices_sorted]
        combo, best_err = _meilleure_combinaison(radii_sorted, candidats)
        fiabilite = _fiabilite_taille(radii_sorted, best_err)

        use_size_as_primary = fiabilite >= 0.45 and len(indices_sorted) >= 2

        for pos, i in enumerate(indices_sorted):
            desc = descriptors[i]
            scores = _score_couleur_intragroupe(desc, groupe)
            if not scores:
                denom = combo[min(pos, len(combo) - 1)]
                conf = desc["group_conf"] * 0.3
                final_denoms[i] = denom
                final_confs[i] = round(conf, 3)
                continue

            if use_size_as_primary and pos < len(combo):
                denom_taille = combo[pos]
                color_support = scores.get(denom_taille, 0.0)
                conf = desc["group_conf"] * (0.45 + 0.4 * fiabilite + 0.15 * color_support)
                final_denoms[i] = denom_taille
                final_confs[i] = round(min(1.0, conf), 3)
                continue

            if fiabilite > 0.0 and pos < len(combo):
                denom_taille = combo[pos]
                scores = dict(scores)
                scores[denom_taille] = scores.get(denom_taille, 1e-6) * (1.0 + fiabilite * 3.2)
                total = sum(scores.values())
                scores = {d: v / total for d, v in scores.items()}

            best_denom = max(scores, key=lambda d: scores[d])
            best_score = scores[best_denom]
            margin = best_score - np.mean([v for d, v in scores.items() if d != best_denom])

            conf = desc["group_conf"] * (0.32 + 0.45 * fiabilite + 0.25 * max(0.0, margin))
            final_denoms[i] = best_denom
            final_confs[i] = round(min(1.0, conf), 3)

    for i, denom in enumerate(final_denoms):
        if denom is None:
            desc = descriptors[i]
            if desc["group"] == "cuivre":
                denom = "2c"
            elif desc["group"] == "or":
                denom = "20c"
            else:
                denom = "1e"
            final_denoms[i] = denom
            final_confs[i] = 0.1

    resultats = []
    for i, circle in enumerate(circles):
        denom = final_denoms[i]
        groupe = next((g for g, ds in _GROUPES.items() if denom in ds), "")
        resultats.append(
            ValeurPiece(
                cercle=circle,
                denomination=denom,
                valeur_centimes=VALEURS_CENTIMES[denom],
                confiance=min(1.0, final_confs[i]),
                groupe_couleur=groupe,
            )
        )

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
