from __future__ import annotations

"""Détermination de la valeur des pièces euro.

Algorithme principal (``classify_by_color_and_size``) — deux passes :

  Passe 1 — Groupe de couleur (sigmoid sur H)
    Cuivré   (1c, 2c, 5c)   : H ≈ 11-14, S élevée
    Or nord. (10c, 20c, 50c): H ≈ 20-26, S élevée
    Bimét.   (1e, 2e)       : S faible globalement
    Les deux frontières cuivré/or sont séparées par un sigmoid centré sur H=17.
    Chaque pièce est normalisée par CLAHE local avant extraction HSV pour
    résister aux images sombres.  Le seuil V est adaptatif (médiane locale).

  Passe 2 — Dénomination au sein du groupe
    Dans chaque groupe, les pièces sont triées par rayon et les dénominations
    candidates par diamètre réel.  La meilleure correspondance est sélectionnée
    par minimisation de l'erreur relative sur les ratios de taille.
    Pour 1e/2e, l'analyse centre/couronne s'ajoute comme critère.
"""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from modules.segmentation import DetectedCircle


# ---------------------------------------------------------------------------
# Tables de référence
# ---------------------------------------------------------------------------

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

# Groupes ordonnés par diamètre croissant
_GROUPES: dict[str, list[str]] = {
    "cuivre":     ["1c", "2c", "5c"],
    "or":         ["10c", "20c", "50c"],
    "bimetallic": ["1e", "2e"],
}

_DENOMINATIONS = list(DIAMETRES_MM.keys())
_DIAMS = np.array([DIAMETRES_MM[d] for d in _DENOMINATIONS], dtype=float)

# H de frontière entre cuivré et or (milieu entre H≈12 et H≈22)
_H_FRONTIERE = 17.0
# Pente du sigmoid de séparation (plus élevée = frontière plus nette)
_SIGMOID_PENTE = 0.55

# Plages HSV pour l'analyse bimétallique
_HSV_OR     = {"h_min": 15, "h_max": 34, "s_min": 60, "v_min": 70}
_HSV_ARGENT = {"s_max": 75, "v_min": 60}


# ---------------------------------------------------------------------------
# Structure de résultat
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValeurPiece:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float          # 0.0–1.0
    groupe_couleur: str = ""  # "cuivre", "or", "bimetallic"


# ---------------------------------------------------------------------------
# Extraction de pixels : CLAHE local + seuil V adaptatif
# ---------------------------------------------------------------------------

def _pixels_hsv_normalises(
    image_bgr: np.ndarray,
    circle: DetectedCircle,
    ratio_ext: float = 0.88,
    ratio_int: float = 0.0,
) -> np.ndarray:
    """Pixels HSV d'une zone annulaire de la pièce, après CLAHE local.

    Le CLAHE est appliqué uniquement sur la ROI de la pièce, ce qui normalise
    la luminosité indépendamment du reste de l'image (robustesse aux images
    sombres ou surexposées).
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    r_ext = max(1, int(r * ratio_ext))

    x1 = max(0, cx - r_ext)
    y1 = max(0, cy - r_ext)
    x2 = min(image_bgr.shape[1], cx + r_ext)
    y2 = min(image_bgr.shape[0], cy + r_ext)

    roi = image_bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # Normalisation CLAHE sur le canal L (LAB)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    roi_norm = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # Masque annulaire dans le repère local
    lx, ly = cx - x1, cy - y1
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), r_ext, 255, -1)
    if ratio_int > 0:
        r_int = max(1, int(r * ratio_int))
        cv2.circle(mask, (lx, ly), r_int, 0, -1)

    hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
    return hsv[mask > 0]


# ---------------------------------------------------------------------------
# Statistiques HSV robustes
# ---------------------------------------------------------------------------

def _stats_hsv(pixels: np.ndarray) -> dict:
    """Statistiques robustes depuis pixels HSV.

    - Seuil V adaptatif (25 % de la médiane) pour images sombres.
    - Teinte pondérée par la saturation (H fiable seulement si S > 35).
    """
    if len(pixels) == 0:
        return {"h": 0.0, "s": 0.0, "couv": 0.0, "fiable": False}

    h = pixels[:, 0].astype(float)
    s = pixels[:, 1].astype(float)
    v = pixels[:, 2].astype(float)

    v_med = float(np.median(v))
    v_bas  = max(20.0, v_med * 0.25)
    v_haut = min(250.0, v_med * 1.9)
    ok = (v >= v_bas) & (v <= v_haut)
    h_v, s_v = h[ok], s[ok]

    if len(h_v) == 0:
        return {"h": 0.0, "s": 0.0, "couv": 0.0, "fiable": False}

    sature = s_v > 35
    couv = float(sature.sum()) / len(h_v)

    if sature.sum() >= 5:
        h_pond = float(np.average(h_v[sature], weights=s_v[sature]))
    else:
        h_pond = float(np.median(h_v))

    return {
        "h": h_pond,
        "s": float(np.mean(s_v)),
        "couv": couv,
        "fiable": couv > 0.10 and len(h_v) > 8,
    }


# ---------------------------------------------------------------------------
# Score de groupe de couleur (sigmoid + Gaussienne)
# ---------------------------------------------------------------------------

def _score_groupe(stats: dict, groupe: str) -> float:
    """Score [0, 1] d'appartenance à un groupe de couleur.

    Cuivré/Or : combinaison d'un sigmoid centré sur H=17 (frontière entre
    les deux groupes) et d'une Gaussienne centrée sur la teinte cible.
    Cette double approche est robuste aux petits décalages de teinte dus à
    l'éclairage tout en maintenant une frontière nette entre groupes.
    Bimétallique : faible saturation globale + faible couverture saturée.
    """
    h   = stats["h"]
    s   = stats["s"]
    cov = stats["couv"]

    if groupe == "cuivre":
        # Sigmoid : 1 si H << 17, 0 si H >> 17
        sig = 1.0 / (1.0 + np.exp(_SIGMOID_PENTE * (h - _H_FRONTIERE)))
        # Gaussienne centrée sur H=12 (cuivre typique)
        gau = float(np.exp(-0.5 * ((h - 12.0) / 5.0) ** 2))
        score_h = 0.55 * sig + 0.45 * gau
        score_s = float(np.clip((s - 45.0) / 65.0, 0.0, 1.0))
        return score_h * max(0.05, score_s)

    if groupe == "or":
        # Sigmoid : 0 si H << 17, 1 si H >> 17
        sig = 1.0 / (1.0 + np.exp(-_SIGMOID_PENTE * (h - _H_FRONTIERE)))
        # Gaussienne centrée sur H=22 (or nordique typique)
        gau = float(np.exp(-0.5 * ((h - 22.0) / 5.0) ** 2))
        score_h = 0.55 * sig + 0.45 * gau
        score_s = float(np.clip((s - 55.0) / 65.0, 0.0, 1.0))
        return score_h * max(0.05, score_s)

    # bimetallic : faible saturation et faible couverture colorée
    score_s   = float(np.clip(1.0 - s / 75.0, 0.0, 1.0))
    score_cov = float(np.clip(1.0 - cov / 0.28, 0.0, 1.0))
    return 0.55 * score_s + 0.45 * score_cov


# ---------------------------------------------------------------------------
# Analyse bimétallique centre / couronne (1e vs 2e)
# ---------------------------------------------------------------------------

def _score_bimetallic(image_bgr: np.ndarray, circle: DetectedCircle) -> tuple[float, float]:
    """Retourne (score_1e, score_2e) normalisés entre 0 et 1.

    1e : centre argenté + couronne dorée.
    2e : centre doré   + couronne argentée.
    """
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


# ---------------------------------------------------------------------------
# Sélection de la meilleure combinaison de dénominations dans un groupe
# ---------------------------------------------------------------------------

def _meilleure_combinaison(radii: list[float], candidats: list[str]) -> list[str]:
    """Sélectionne N dénominations parmi M candidats (N ≤ M ≤ 3).

    Minimise la somme des erreurs relatives sur tous les ratios pairwise.
    Si N > M, les pièces en excès sont assignées au candidat extrême le plus
    proche (plusieurs pièces peuvent recevoir la même dénomination).

    Args:
        radii:     rayons triés par ordre croissant.
        candidats: dénominations du groupe, triées par diamètre croissant.
    """
    n = len(radii)
    m = len(candidats)

    if n == 0:
        return []

    # N > M : assigner chaque pièce au candidat dont le diamètre ratio est le plus proche
    if n > m:
        ref_r = radii[0]
        ref_d = DIAMETRES_MM[candidats[0]]
        result = []
        for r in radii:
            ratio_obs = r / ref_r
            best = min(candidats, key=lambda d: abs(DIAMETRES_MM[d] / ref_d - ratio_obs))
            result.append(best)
        return result

    # N == M : assignation directe dans l'ordre (taille croissante → dénomination croissante)
    if n == m:
        return list(candidats)

    # N < M : choisir N candidats parmi M qui minimisent l'erreur de ratio
    if n == 1:
        # Impossible de distinguer par ratio : choisir le candidat médian du groupe
        return [candidats[m // 2]]

    # n >= 2 et n < m : essayer toutes les combinaisons C(m, n)
    best_combo: list[str] = list(candidats[:n])
    best_err = float("inf")

    for combo in combinations(candidats, n):
        diams = [DIAMETRES_MM[d] for d in combo]
        err = 0.0
        for a in range(n):
            for b in range(a + 1, n):
                ratio_obs = radii[b] / radii[a]
                ratio_exp = diams[b] / diams[a]
                err += abs(ratio_obs - ratio_exp) / ratio_exp
        if err < best_err:
            best_err = err
            best_combo = list(combo)

    return best_combo


# ---------------------------------------------------------------------------
# Stratégie 1 : taille relative uniquement
# ---------------------------------------------------------------------------

def classify_by_relative_size(circles: list[DetectedCircle]) -> list[ValeurPiece]:
    """Détermine la valeur des pièces uniquement par taille relative."""
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


# ---------------------------------------------------------------------------
# Stratégie 2 : taille absolue avec pièce de référence
# ---------------------------------------------------------------------------

def classify_with_reference(
    circles: list[DetectedCircle],
    cercle_reference: DetectedCircle,
    denomination_reference: str,
) -> list[ValeurPiece]:
    """Détermine la valeur des pièces via une pièce de référence connue."""
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


# ---------------------------------------------------------------------------
# Stratégie 3 : deux passes couleur → taille (recommandée)
# ---------------------------------------------------------------------------

def classify_by_color_and_size(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPiece]:
    """Détermine la valeur des pièces en deux passes.

    **Passe 1 — groupe de couleur (sigmoid + Gaussienne)**
    Pour chaque pièce, calcule un score d'appartenance à chacun des trois
    groupes (cuivré, or, bimétallique).  La discrimination cuivré/or repose
    sur un sigmoid centré sur H=17 combiné à une Gaussienne sur la teinte
    cible, ce qui résiste aux petits décalages dus à l'éclairage.
    Un CLAHE local est appliqué avant l'extraction HSV pour normaliser la
    luminosité de chaque pièce indépendamment.

    **Passe 2 — dénomination dans le groupe (ratios de taille)**
    Dans chaque groupe, les pièces sont triées par rayon et les candidats
    par diamètre réel.  La combinaison (N parmi 3) qui minimise l'erreur
    relative sur tous les ratios pairwise est retenue.  Cette approche ne
    requiert pas d'étalonnage absolu.
    Pour le groupe bimétallique, l'analyse centre/couronne est combinée à
    la taille pour distinguer 1e (23,25 mm) de 2e (25,75 mm).
    """
    if not circles:
        return []

    n = len(circles)

    # --- Passe 1 : extraire les stats HSV et scorer les groupes ---
    all_stats = [
        _stats_hsv(_pixels_hsv_normalises(image_bgr, c, ratio_ext=0.85))
        for c in circles
    ]

    groupes_ordre = list(_GROUPES.keys())  # ["cuivre", "or", "bimetallic"]
    groupe_scores = np.array([
        [_score_groupe(st, g) for g in groupes_ordre]
        for st in all_stats
    ])  # shape (n, 3)

    # Normaliser par ligne pour obtenir des probabilités
    totaux = groupe_scores.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1.0
    groupe_proba = groupe_scores / totaux

    assigned_groupes = [groupes_ordre[int(np.argmax(groupe_proba[i]))] for i in range(n)]

    # --- Analyse bimétallique systématique ---
    bi_scores = [_score_bimetallic(image_bgr, c) for c in circles]

    # --- Passe 2 : dénomination dans le groupe ---
    final_denoms: list[str | None] = [None] * n
    final_confs:  list[float]       = [0.0]  * n

    for groupe in groupes_ordre:
        candidats = _GROUPES[groupe]  # triés par diamètre croissant
        indices = [i for i in range(n) if assigned_groupes[i] == groupe]

        if not indices:
            continue

        conf_groupe = float(np.mean([groupe_proba[i, groupes_ordre.index(groupe)] for i in indices]))

        if groupe == "bimetallic":
            # Utiliser centre/couronne + taille pour 1e vs 2e
            for i in indices:
                s1e, s2e = bi_scores[i]
                # Taille : 2e est toujours plus grande que 1e
                # Bonus taille si plusieurs bimétalliques
                if len(indices) >= 2:
                    autres = [circles[j].radius for j in indices if j != i]
                    r_i = circles[i].radius
                    # Si cette pièce est la plus grande du groupe bimétallique → 2e
                    if all(r_i >= r_j for r_j in autres):
                        s2e = min(1.0, s2e + 0.20)
                    else:
                        s1e = min(1.0, s1e + 0.20)

                denom = "2e" if s2e > s1e else "1e"
                final_denoms[i] = denom
                final_confs[i]  = round(conf_groupe * max(s1e, s2e), 3)

        else:
            # Trier par rayon croissant
            indices_tries = sorted(indices, key=lambda i: circles[i].radius)
            radii_tries   = [float(circles[i].radius) for i in indices_tries]

            # Trouver la meilleure combinaison de dénominations
            combo = _meilleure_combinaison(radii_tries, candidats)

            for i, denom in zip(indices_tries, combo):
                # Confiance = score du groupe × cohérence de taille (ratio moyen)
                diam_cible = DIAMETRES_MM[denom]
                if len(indices_tries) >= 2:
                    ref_r = radii_tries[0]
                    ref_d = DIAMETRES_MM[combo[0]]
                    ratio_obs = circles[i].radius / ref_r
                    ratio_exp = diam_cible / ref_d
                    coherence = max(0.0, 1.0 - abs(ratio_obs - ratio_exp) / max(ratio_exp, 0.01) / 0.12)
                else:
                    coherence = 0.6  # Coin unique dans le groupe : confiance modérée
                final_denoms[i] = denom
                final_confs[i]  = round(conf_groupe * coherence, 3)

    # --- Fallback pour les pièces non assignées (ne devrait pas arriver) ---
    for i in range(n):
        if final_denoms[i] is None:
            final_denoms[i] = _DENOMINATIONS[int(np.argmax(groupe_scores[i]))]
            final_confs[i]  = 0.1

    # --- Construction des résultats ---
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


# ---------------------------------------------------------------------------
# Calcul du total
# ---------------------------------------------------------------------------

def valeur_totale(valuations: list[ValeurPiece]) -> tuple[int, str]:
    """Somme les pièces et retourne ``(centimes, libellé)`` ex. ``"3e45"``."""
    total = sum(v.valeur_centimes for v in valuations)
    euros, centimes = divmod(total, 100)
    if euros > 0 and centimes > 0:
        libelle = f"{euros}e{centimes:02d}"
    elif euros > 0:
        libelle = f"{euros}e"
    else:
        libelle = f"{total}c"
    return total, libelle
