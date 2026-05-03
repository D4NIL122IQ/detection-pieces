from __future__ import annotations

"""Classification des pièces d'euros par filtres d'image.

Approche alternative : au lieu d'analyser directement la couleur HSV/HLS,
on applique des filtres (inversion, seuillage, différence de canaux) pour
séparer les groupes et différencier les valeurs au sein d'un groupe.

Principes :
- Filtre R-B (rouge minus bleu) : sépare cuivre (R-B élevé) de or (R-B moyen) de bimétal (R-B faible)
- Inversion + canal vert : amplifie la différence cuivre/or
- Filtre Laplacien : la variance des bords (texture) diffère par dénomination
- Rapport centre/bord en niveaux de gris : distingue les bimétaux
"""

from dataclasses import dataclass

import cv2
import numpy as np

from modules.segmentation import DetectedCircle, apply_clahe_bgr

DIAMETRES_MM: dict[str, float] = {
    "1c": 16.25, "2c": 18.75, "5c": 21.25,
    "10c": 19.75, "20c": 22.25, "50c": 24.25,
    "1e": 23.25, "2e": 25.75,
}

VALEURS_CENTIMES: dict[str, int] = {
    "1c": 1, "2c": 2, "5c": 5,
    "10c": 10, "20c": 20, "50c": 50,
    "1e": 100, "2e": 200,
}

_GROUPES: dict[str, list[str]] = {
    "cuivre": ["1c", "2c", "5c"],
    "or": ["10c", "20c", "50c"],
    "bimetallic": ["1e", "2e"],
}


@dataclass(frozen=True)
class ValeurPieceFiltre:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""


def _extract_roi(image_bgr: np.ndarray, circle: DetectedCircle, ratio: float = 0.85) -> np.ndarray:
    """Extrait la ROI circulaire normalisée à 64x64."""
    cx, cy, r = circle.x, circle.y, circle.radius
    r_ext = max(1, int(r * ratio))

    x1 = max(0, cx - r_ext)
    y1 = max(0, cy - r_ext)
    x2 = min(image_bgr.shape[1], cx + r_ext)
    y2 = min(image_bgr.shape[0], cy + r_ext)

    roi = image_bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    # Masquer les pixels hors du cercle
    lx, ly = cx - x1, cy - y1
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), r_ext, 255, -1)
    roi[mask == 0] = 0

    # Redimensionner à 64x64
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    return roi


def _filtre_ratios_canaux(roi: np.ndarray) -> dict[str, float]:
    """Calcule les ratios normalisés R/(R+G+B), G/(R+G+B), B/(R+G+B).

    Ces ratios sont invariants à la luminosité (robustes aux conditions d'éclairage).
    """
    b, g, r = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)
    mask = (b + g + r) > 30
    if mask.sum() < 10:
        return {"r_ratio": 0.333, "g_ratio": 0.333, "b_ratio": 0.333}

    total = b[mask] + g[mask] + r[mask] + 1e-6
    return {
        "r_ratio": float(np.mean(r[mask] / total)),
        "g_ratio": float(np.mean(g[mask] / total)),
        "b_ratio": float(np.mean(b[mask] / total)),
    }


def _filtre_rb(roi: np.ndarray) -> float:
    """Filtre Rouge - Bleu (conservé pour compatibilité intra-groupe)."""
    b, g, r = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)
    mask = (b + g + r) > 30
    if mask.sum() < 10:
        return 0.0
    return float(np.mean(r[mask] - b[mask]))


def _filtre_rg(roi: np.ndarray) -> float:
    """Filtre Rouge - Vert."""
    b, g, r = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)
    mask = (b + g + r) > 30
    if mask.sum() < 10:
        return 0.0
    return float(np.mean(r[mask] - g[mask]))


def _filtre_laplacien_var(roi: np.ndarray) -> float:
    """Variance du Laplacien (mesure de texture/détail). Varie par dénomination."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mask = gray > 10
    if mask.sum() < 10:
        return 0.0
    return float(np.var(lap[mask]))


def _filtre_contraste_centre_bord(image_bgr: np.ndarray, circle: DetectedCircle) -> float:
    """Différence de luminosité entre centre et couronne — utile pour bimétaux."""
    cx, cy, r = circle.x, circle.y, circle.radius
    r_centre = max(1, int(r * 0.45))
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

    mask_centre = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_centre, (lx, ly), r_centre, 255, -1)

    mask_bord = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_bord, (lx, ly), r_ext, 255, -1)
    cv2.circle(mask_bord, (lx, ly), int(r * 0.52), 0, -1)

    moy_centre = float(np.mean(gray[mask_centre > 0])) if mask_centre.sum() > 0 else 0.0
    moy_bord = float(np.mean(gray[mask_bord > 0])) if mask_bord.sum() > 0 else 0.0

    return moy_centre - moy_bord


def _classify_groupe(features: dict) -> tuple[str, float]:
    """Détermine le groupe à partir des ratios de canaux normalisés.

    Calibré sur données réelles (ratios R/G/B normalisés) :
      Cuivre :    R=0.429, G=0.322, B=0.248
      Or :        R=0.391, G=0.354, B=0.255
      Bimétal :   R=0.349, G=0.336, B=0.315

    Signaux discriminants :
      - R_ratio : cuivre > or > bimétal
      - B_ratio : bimétal >> cuivre ≈ or
      - G_ratio : or > bimétal > cuivre
    """
    r_ratio = features["r_ratio"]
    g_ratio = features["g_ratio"]
    b_ratio = features["b_ratio"]

    # Scores par canal (somme pondérée, pas produit — plus tolérant)
    # R_ratio est le signal le plus discriminant, B_ratio le second
    score_cuivre = (
        float(np.exp(-0.5 * ((r_ratio - 0.429) / 0.06) ** 2)) * 0.45 +
        float(np.exp(-0.5 * ((g_ratio - 0.322) / 0.03) ** 2)) * 0.20 +
        float(np.exp(-0.5 * ((b_ratio - 0.248) / 0.06) ** 2)) * 0.35
    )

    score_or = (
        float(np.exp(-0.5 * ((r_ratio - 0.391) / 0.04) ** 2)) * 0.40 +
        float(np.exp(-0.5 * ((g_ratio - 0.354) / 0.03) ** 2)) * 0.25 +
        float(np.exp(-0.5 * ((b_ratio - 0.255) / 0.05) ** 2)) * 0.35
    )

    score_bimetallic = (
        float(np.exp(-0.5 * ((r_ratio - 0.349) / 0.05) ** 2)) * 0.35 +
        float(np.exp(-0.5 * ((g_ratio - 0.336) / 0.03) ** 2)) * 0.20 +
        float(np.exp(-0.5 * ((b_ratio - 0.315) / 0.05) ** 2)) * 0.45
    )

    scores = {
        "cuivre": max(score_cuivre, 1e-10),
        "or": max(score_or, 1e-10),
        "bimetallic": max(score_bimetallic, 1e-10),
    }
    best = max(scores, key=lambda g: scores[g])
    total = sum(scores.values())
    conf = scores[best] / total if total > 0 else 0.33
    return best, conf


def _classify_intra_cuivre(features: dict, radius: float, all_radii: list[float]) -> tuple[str, float]:
    """Différencie 1c/2c/5c par filtres + taille relative.

    Diamètres réels : 1c=16.25mm, 2c=18.75mm, 5c=21.25mm
    Ratios : 1c/5c=0.76, 2c/5c=0.88
    La taille relative est le signal le plus fiable ici.
    """
    rb = features["rb"]
    rg = features["rg"]

    # Score couleur (faible discrimination, sigma large)
    score_1c = float(np.exp(-0.5 * ((rb - 60) / 25.0) ** 2)) * float(np.exp(-0.5 * ((rg - 35) / 20.0) ** 2))
    score_2c = float(np.exp(-0.5 * ((rb - 55) / 25.0) ** 2)) * float(np.exp(-0.5 * ((rg - 30) / 20.0) ** 2))
    score_5c = float(np.exp(-0.5 * ((rb - 48) / 25.0) ** 2)) * float(np.exp(-0.5 * ((rg - 25) / 20.0) ** 2))

    scores = {"1c": max(score_1c, 1e-6), "2c": max(score_2c, 1e-6), "5c": max(score_5c, 1e-6)}

    # Taille relative : signal dominant
    if len(all_radii) > 1:
        sorted_r = sorted(all_radii)
        rank = sorted_r.index(radius) / (len(sorted_r) - 1)
        scores["1c"] *= (1.0 + 3.0 * (1.0 - rank))
        scores["5c"] *= (1.0 + 3.0 * rank)
        scores["2c"] *= (1.0 + 1.5 * (1.0 - abs(rank - 0.5) * 2))

    total = sum(scores.values())
    scores = {d: v / total for d, v in scores.items()}
    best = max(scores, key=lambda d: scores[d])
    return best, scores[best]


def _classify_intra_or(features: dict, radius: float, all_radii: list[float]) -> tuple[str, float]:
    """Différencie 10c/20c/50c par filtres + taille relative.

    Diamètres réels : 10c=19.75mm, 20c=22.25mm, 50c=24.25mm
    La taille relative est le signal le plus fiable.
    """
    rb = features["rb"]

    # Score couleur (discrimination faible entre or nordique)
    score_10c = float(np.exp(-0.5 * ((rb - 33) / 12.0) ** 2))
    score_20c = float(np.exp(-0.5 * ((rb - 39) / 12.0) ** 2))
    score_50c = float(np.exp(-0.5 * ((rb - 45) / 12.0) ** 2))

    scores = {"10c": max(score_10c, 1e-6), "20c": max(score_20c, 1e-6), "50c": max(score_50c, 1e-6)}

    # Taille relative : signal dominant
    if len(all_radii) > 1:
        sorted_r = sorted(all_radii)
        rank = sorted_r.index(radius) / (len(sorted_r) - 1)
        scores["10c"] *= (1.0 + 3.0 * (1.0 - rank))
        scores["50c"] *= (1.0 + 3.0 * rank)
        scores["20c"] *= (1.0 + 1.5 * (1.0 - abs(rank - 0.5) * 2))

    total = sum(scores.values())
    scores = {d: v / total for d, v in scores.items()}
    best = max(scores, key=lambda d: scores[d])
    return best, scores[best]


def _classify_intra_bimetallic(features: dict, image_bgr: np.ndarray, circle: DetectedCircle,
                                all_radii: list[float]) -> tuple[str, float]:
    """Différencie 1€/2€ par contraste centre/bord + taille."""
    contraste = features["contraste_cb"]

    # 1€ : centre argent (clair) + couronne or (sombre) → contraste > 0
    # 2€ : centre or (sombre) + couronne argent (clair) → contraste < 0
    score_1e = float(np.clip(0.5 + contraste / 40.0, 0.1, 0.9))
    score_2e = float(np.clip(0.5 - contraste / 40.0, 0.1, 0.9))

    # Bonus taille : 2€ est plus grand que 1€
    if len(all_radii) > 1:
        sorted_r = sorted(all_radii)
        rank = sorted_r.index(circle.radius) / (len(sorted_r) - 1)
        score_2e *= (1.0 + 0.4 * rank)
        score_1e *= (1.0 + 0.4 * (1.0 - rank))

    total = score_1e + score_2e
    if score_2e > score_1e:
        return "2e", score_2e / total
    return "1e", score_1e / total


def classify_filtres(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPieceFiltre]:
    """Classification par filtres d'image."""
    if not circles:
        return []

    n = len(circles)

    # Extraire les features pour chaque pièce
    all_features = []
    for c in circles:
        roi = _extract_roi(image_bgr, c)
        ratios = _filtre_ratios_canaux(roi)
        features = {
            "r_ratio": ratios["r_ratio"],
            "g_ratio": ratios["g_ratio"],
            "b_ratio": ratios["b_ratio"],
            "rb": _filtre_rb(roi),
            "rg": _filtre_rg(roi),
            "laplacien": _filtre_laplacien_var(roi),
            "contraste_cb": _filtre_contraste_centre_bord(image_bgr, c),
        }
        all_features.append(features)

    # Classification par groupe
    groupes = []
    confs_groupe = []
    for feat in all_features:
        g, conf = _classify_groupe(feat)
        groupes.append(g)
        confs_groupe.append(conf)

    # Classification intra-groupe
    resultats = []
    for groupe in _GROUPES:
        indices = [i for i in range(n) if groupes[i] == groupe]
        if not indices:
            continue

        radii_groupe = [float(circles[i].radius) for i in indices]

        for i in indices:
            feat = all_features[i]
            r = float(circles[i].radius)

            if groupe == "cuivre":
                denom, conf_intra = _classify_intra_cuivre(feat, r, radii_groupe)
            elif groupe == "or":
                denom, conf_intra = _classify_intra_or(feat, r, radii_groupe)
            else:
                denom, conf_intra = _classify_intra_bimetallic(feat, image_bgr, circles[i], radii_groupe)

            conf_finale = round(confs_groupe[i] * conf_intra, 3)
            resultats.append((i, denom, conf_finale, groupe))

    # Trier par index original
    resultats.sort(key=lambda x: x[0])

    return [
        ValeurPieceFiltre(
            cercle=circles[i],
            denomination=denom,
            valeur_centimes=VALEURS_CENTIMES[denom],
            confiance=min(1.0, conf),
            groupe_couleur=groupe,
        )
        for i, denom, conf, groupe in resultats
    ]


def valeur_totale_filtres(valuations: list[ValeurPieceFiltre]) -> tuple[int, str]:
    total = sum(v.valeur_centimes for v in valuations)
    euros, centimes = divmod(total, 100)
    if euros > 0 and centimes > 0:
        libelle = f"{euros}e{centimes:02d}"
    elif euros > 0:
        libelle = f"{euros}e"
    else:
        libelle = f"{total}c"
    return total, libelle
