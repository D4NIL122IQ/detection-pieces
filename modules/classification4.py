from __future__ import annotations

"""Classification combinée : ensemble de HSV + Filtres + contraste centre/bord.

Combine les résultats de classification.py (HSV) et classification3.py (Filtres RGB)
par vote pondéré sur le groupe, avec un détecteur bimétal basé sur la différence
de couleur entre le centre et la couronne de la pièce.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from itertools import product as iter_product

from modules.constants import VALEURS_CENTIMES, GROUPES, DIAMETRES_MM
from modules.segmentation import DetectedCircle
from modules.classification import classify_by_color_and_size
from modules.classification3 import classify_filtres


_DENOM_TO_GROUPE = {d: g for g, ds in GROUPES.items() for d in ds}


@dataclass(frozen=True)
class ValeurPieceCombinee:
    cercle: DetectedCircle
    denomination: str
    valeur_centimes: int
    confiance: float
    groupe_couleur: str = ""
    methode_dominante: str = ""


def _hue_circular_std(image_bgr: np.ndarray, circle: DetectedCircle) -> float:
    """Écart-type circulaire de la teinte HSV sur le disque de la pièce.

    Un bimétal présente deux zones de couleur distinctes (argent + or) →
    distribution bimodale de teinte → écart-type circulaire élevé (>30°).
    Un monométal a une couleur uniforme → écart-type faible (<12°).

    Retourne une valeur en degrés (0 = couleur parfaitement uniforme).
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    if r < 10:
        return 0.0
    h, w = image_bgr.shape[:2]
    r_in = max(1, int(r * 0.88))
    x1, y1 = max(0, cx - r_in), max(0, cy - r_in)
    x2, y2 = min(w, cx + r_in), min(h, cy + r_in)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    lx, ly = cx - x1, cy - y1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), r_in, 255, -1)
    h_vals = hsv[:, :, 0][mask > 0].astype(np.float64)
    s_vals = hsv[:, :, 1][mask > 0].astype(np.float64)
    # Exclure les pixels peu saturés (gris/noir) dont la teinte est instable
    sat_mask = s_vals > 35
    if sat_mask.sum() < 30:
        return 0.0
    h_sat = h_vals[sat_mask]
    # Variance circulaire : angles en [0, 2π], H OpenCV ∈ [0, 180] → *π/90
    angles = h_sat * (np.pi / 90.0)
    c_m = float(np.mean(np.cos(angles)))
    s_m = float(np.mean(np.sin(angles)))
    r_val = min(np.sqrt(c_m ** 2 + s_m ** 2), 1.0 - 1e-9)
    # Écart-type circulaire en degrés
    return float(np.sqrt(-2.0 * np.log(r_val)) * (90.0 / np.pi))


def _bimetal_kmeans_score(
    image_bgr: np.ndarray,
    circle: DetectedCircle,
) -> float:
    """Score bimétal par séparation des clusters K-means (k=2) sur LAB.

    Bimétal → 2 clusters bien séparés (argent et or).
    Monométal → 2 clusters proches (variations d'ombre).

    Retourne un score entre 0 et 1 basé sur la distance LAB inter-clusters
    normalisée.
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    if r < 12:
        return 0.0

    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    lx, ly = cx - x1, cy - y1
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (lx, ly), int(r * 0.92), 255, -1)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0].astype(np.float32)
    if len(pixels) < 50:
        return 0.0

    # Sous-échantillonner pour la vitesse
    if len(pixels) > 2000:
        idx = np.random.choice(len(pixels), 2000, replace=False)
        pixels = pixels[idx]

    # K-means k=2 sur les canaux a et b (chrominance, ignore luminance)
    ab = pixels[:, 1:3]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, _, centers = cv2.kmeans(
            ab, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
        )
    except cv2.error:
        return 0.0

    # Distance entre les 2 centres (en chrominance LAB)
    sep = float(np.linalg.norm(centers[0] - centers[1]))

    # Bimétal typique : séparation ~8-15 ; monométal : ~2-5
    return float(np.clip(sep / 12.0, 0.0, 1.0))


def _bimetal_radial_profile_score(
    image_bgr: np.ndarray,
    circle: DetectedCircle,
    n_rings: int = 20,
) -> tuple[float, str]:
    """Détecte un bimétal par profil radial de saturation.

    Au lieu de comparer centre vs couronne globalement, on trace la saturation
    moyenne sur 20 anneaux concentriques de 0 à r. Un bimétal présente une
    discontinuité abrupte (saut) ; un monométal a un profil lisse.

    Retourne (score, denomination) avec score = amplitude maximale du saut
    normalisé, denomination basée sur quel côté du saut est saturé.
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    if r < 12:
        return 0.0, "1e"

    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, "1e"

    lx, ly = cx - x1, cy - y1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)

    # Carte des distances au centre
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)

    # Saturation moyenne par anneau (de 0 à r, n_rings anneaux)
    profile = np.zeros(n_rings, dtype=np.float32)
    counts = np.zeros(n_rings, dtype=np.int32)
    bins = (dist / max(r, 1) * n_rings).astype(np.int32)
    bins = np.clip(bins, 0, n_rings - 1)

    flat_bins = bins.ravel()
    flat_s = s_channel.ravel()
    for i in range(n_rings):
        m = flat_bins == i
        if m.sum() > 0:
            profile[i] = flat_s[m].mean()
            counts[i] = m.sum()

    if (counts > 0).sum() < n_rings - 2:
        return 0.0, "1e"

    # Lissage léger pour réduire le bruit
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    smooth = np.convolve(profile, kernel, mode="same")

    # On ignore les 2 anneaux extrêmes (bord = artéfacts) → indices 2..n_rings-3
    inner = smooth[2:n_rings - 2]
    if len(inner) < 5:
        return 0.0, "1e"

    # Recherche du saut maximal (différence entre 2 moitiés autour d'un seuil)
    best_jump = 0.0
    best_split = len(inner) // 2
    for split in range(2, len(inner) - 2):
        left_mean = inner[:split].mean()
        right_mean = inner[split:].mean()
        jump = abs(left_mean - right_mean)
        if jump > best_jump:
            best_jump = jump
            best_split = split

    # Score normalisé : un bimétal a typiquement un saut de 30-60 en saturation
    score = float(np.clip(best_jump / 40.0, 0.0, 1.0))

    # 1€ = centre argent (S bas) puis couronne or (S haut) → left < right
    # 2€ = centre or (S haut) puis couronne argent (S bas) → left > right
    left_mean = inner[:best_split].mean()
    right_mean = inner[best_split:].mean()
    denom = "1e" if left_mean < right_mean else "2e"

    return score, denom


def _bimetal_center_ring_score(
    image_bgr: np.ndarray,
    circle: DetectedCircle,
) -> tuple[float, str]:
    """Détecte si une pièce est bimétallique par contraste couleur centre vs couronne.

    Les pièces cuivre et or ont une couleur uniforme sur toute la surface.
    Les bimétaux ont deux zones distinctes :
      - 1€ : centre argenté (faible saturation) + couronne dorée (saturation élevée)
      - 2€ : centre doré (saturation élevée) + couronne argentée (faible saturation)

    Retourne (score_bimetal, denomination_suggeree) :
      - score entre 0 (uniforme = pas bimétal) et 1 (contraste fort = bimétal)
      - "1e" ou "2e" selon la disposition des couleurs
    """
    cx, cy, r = circle.x, circle.y, circle.radius
    if r < 10:
        return 0.0, "1e"

    h, w = image_bgr.shape[:2]
    r_center = max(1, int(r * 0.42))
    r_ring_in = max(1, int(r * 0.52))
    r_ring_out = max(1, int(r * 0.85))

    # Extraire la ROI
    x1 = max(0, cx - r_ring_out)
    y1 = max(0, cy - r_ring_out)
    x2 = min(w, cx + r_ring_out)
    y2 = min(h, cy + r_ring_out)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, "1e"

    lx, ly = cx - x1, cy - y1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Masque centre (disque intérieur)
    mask_center = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask_center, (lx, ly), r_center, 255, -1)

    # Masque couronne (anneau extérieur)
    mask_ring = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask_ring, (lx, ly), r_ring_out, 255, -1)
    cv2.circle(mask_ring, (lx, ly), r_ring_in, 0, -1)

    px_center = hsv[mask_center > 0]
    px_ring = hsv[mask_ring > 0]

    if len(px_center) < 20 or len(px_ring) < 20:
        return 0.0, "1e"

    # Saturation moyenne de chaque zone
    s_center = float(np.mean(px_center[:, 1]))
    s_ring = float(np.mean(px_ring[:, 1]))

    # Teinte moyenne pondérée par saturation (ignore les pixels gris)
    def weighted_hue(px: np.ndarray) -> float:
        h_vals = px[:, 0].astype(float)
        s_vals = np.maximum(px[:, 1].astype(float), 1.0)
        angles = h_vals * (2.0 * np.pi / 180.0)
        x = float(np.sum(np.cos(angles) * s_vals))
        y = float(np.sum(np.sin(angles) * s_vals))
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += 2.0 * np.pi
        return float(angle * (180.0 / (2.0 * np.pi)))

    h_center = weighted_hue(px_center)
    h_ring = weighted_hue(px_ring)

    # Différence de saturation : signal principal
    # Bimétal = une zone saturée (dorée) et une zone désaturée (argentée)
    delta_s = abs(s_center - s_ring)

    # Différence de teinte : signal complémentaire
    delta_h = min(abs(h_center - h_ring), 180.0 - abs(h_center - h_ring))

    # Score bimétal : contraste de saturation fort + un peu de contraste de teinte
    score = float(np.clip(delta_s / 35.0, 0.0, 0.8)) + float(np.clip(delta_h / 25.0, 0.0, 0.2))
    score = float(np.clip(score, 0.0, 1.0))

    # Déterminer 1€ vs 2€
    # 1€ : centre argent (S bas) + couronne or (S haut) → s_center < s_ring
    # 2€ : centre or (S haut) + couronne argent (S bas) → s_center > s_ring
    denom = "2e" if s_center > s_ring else "1e"

    return score, denom


def classify_combine(
    circles: list[DetectedCircle],
    image_bgr: np.ndarray,
) -> list[ValeurPieceCombinee]:
    """Classification combinée par vote pondéré + détection bimétal centre/bord.

    Stratégie :
    1. Obtenir les résultats HSV et Filtres indépendamment
    2. Calculer un score bimétal par contraste centre/couronne
    3. Voter sur le groupe (HSV + Filtres + signal bimétal)
    4. Prendre la meilleure valeur dans le groupe élu
    """
    if not circles:
        return []

    # Note : l'égalisation a été testée pour Filtres et bimétal mais dégrade
    # les résultats (l'equalizeHist sur L uniformise la luminance et tue le
    # contraste de saturation). HSV fait sa propre égalisation en interne.
    results_hsv = classify_by_color_and_size(circles, image_bgr)
    results_filtres = classify_filtres(circles, image_bgr)

    # Pré-calculer le score bimétal pour chaque pièce
    bimetal_scores = [_bimetal_radial_profile_score(image_bgr, c) for c in circles]
    # K-means k=2 : score complémentaire de séparation des couleurs
    kmeans_scores = [_bimetal_kmeans_score(image_bgr, c) for c in circles]
    # Fusion : score combiné = 0.8 * radial + 0.2 * kmeans
    bimetal_scores = [
        (0.8 * s[0] + 0.2 * km, s[1])
        for s, km in zip(bimetal_scores, kmeans_scores)
    ]
    # Écart-type circulaire de teinte : signal bimétal complémentaire
    hue_std_scores = [_hue_circular_std(image_bgr, c) for c in circles]

    n = len(circles)
    resultats = []

    for i in range(n):
        v_hsv = results_hsv[i] if i < len(results_hsv) else None
        v_filt = results_filtres[i] if i < len(results_filtres) else None

        groupe_hsv = v_hsv.groupe_couleur if v_hsv else ""
        groupe_filt = v_filt.groupe_couleur if v_filt else ""
        conf_hsv = v_hsv.confiance if v_hsv else 0.0
        conf_filt = v_filt.confiance if v_filt else 0.0

        bi_score, bi_denom = bimetal_scores[i]
        h_std = hue_std_scores[i]

        # Vote pondéré sur le groupe
        group_votes: dict[str, float] = {}
        if v_hsv and groupe_hsv:
            group_votes[groupe_hsv] = group_votes.get(groupe_hsv, 0.0) + conf_hsv
        if v_filt and groupe_filt:
            group_votes[groupe_filt] = group_votes.get(groupe_filt, 0.0) + conf_filt

        # Signal 1 : score bimétal radial + K-means
        if bi_score > 0.35:
            group_votes["bimetallic"] = group_votes.get("bimetallic", 0.0) + bi_score
        elif bi_score < 0.15:
            # Pénalité explicite — l'ancienne logique (* 0.5) ne fonctionnait
            # pas quand bimetallic n'était pas encore dans group_votes (0.0 * 0.5 = 0)
            group_votes["bimetallic"] = group_votes.get("bimetallic", 0.0) - 0.18

        # Signal 2 : écart-type circulaire de teinte (grand = deux couleurs distinctes)
        # Seuil positif élevé (38°) pour ne pas confondre avec les pièces or texturées
        if h_std > 38.0:
            group_votes["bimetallic"] = group_votes.get("bimetallic", 0.0) + min(0.35, h_std / 100.0)
        elif h_std < 10.0:
            group_votes["bimetallic"] = group_votes.get("bimetallic", 0.0) - 0.08

        if group_votes:
            groupe_elu = max(group_votes, key=lambda g: group_votes[g])
        else:
            groupe_elu = "or"

        # Sélectionner la meilleure valeur dans le groupe élu
        candidats = []
        if v_hsv and groupe_hsv == groupe_elu:
            candidats.append((v_hsv.denomination, conf_hsv, "HSV"))
        if v_filt and groupe_filt == groupe_elu:
            val_boost = 1.3 if groupe_elu == "cuivre" else 1.0
            candidats.append((v_filt.denomination, conf_filt * val_boost, "Filtres"))

        # Pour bimétal élu par le score centre/bord, utiliser sa suggestion
        if groupe_elu == "bimetallic" and bi_score > 0.35:
            candidats.append((bi_denom, bi_score * 0.8, "CentreBord"))

        if not candidats:
            if conf_hsv >= conf_filt and v_hsv:
                candidats.append((v_hsv.denomination, conf_hsv, "HSV"))
            elif v_filt:
                candidats.append((v_filt.denomination, conf_filt, "Filtres"))

        if candidats:
            best_denom, best_conf, methode = max(candidats, key=lambda c: c[1])
        else:
            best_denom, best_conf, methode = "1e", 0.1, "fallback"

        resultats.append(ValeurPieceCombinee(
            cercle=circles[i],
            denomination=best_denom,
            valeur_centimes=VALEURS_CENTIMES[best_denom],
            confiance=round(min(1.0, best_conf), 3),
            groupe_couleur=_DENOM_TO_GROUPE.get(best_denom, ""),
            methode_dominante=methode,
        ))

    # Post-traitement : cohérence inter-pièces par groupe
    resultats = _enforce_group_consistency(resultats)
    return resultats


def _enforce_group_consistency(
    resultats: list[ValeurPieceCombinee],
) -> list[ValeurPieceCombinee]:
    """Ré-assigne les valeurs au sein de chaque groupe par cohérence des ratios.

    Pour chaque groupe (cuivre/or/bimétal) contenant >= 2 pièces dans une image,
    on cherche l'assignation de valeurs qui minimise l'erreur entre les ratios
    de rayons observés et les ratios de diamètres théoriques.

    Pour N pièces : énumération des combinaisons avec répétition possible
    (une image peut contenir 2× la même valeur). Pour N >= 5 on garde
    l'assignation actuelle pour limiter le coût.
    """
    if len(resultats) < 2:
        return resultats

    # Indexer par groupe
    par_groupe: dict[str, list[int]] = {}
    for idx, r in enumerate(resultats):
        par_groupe.setdefault(r.groupe_couleur, []).append(idx)

    out = list(resultats)
    for groupe, indices in par_groupe.items():
        if groupe not in GROUPES or len(indices) < 2 or len(indices) > 5:
            continue
        denoms = GROUPES[groupe]
        if len(denoms) < 2:
            continue

        # Trier par rayon croissant
        indices_sorted = sorted(indices, key=lambda i: resultats[i].cercle.radius)
        rayons = [resultats[i].cercle.radius for i in indices_sorted]

        # Confiance moyenne actuelle (sera comparée à la confiance après ré-assignation)
        conf_moyenne = sum(resultats[i].confiance for i in indices_sorted) / len(indices_sorted)
        # Si confiance déjà élevée, on ne modifie pas
        if conf_moyenne > 0.75:
            continue

        # Énumérer toutes les combinaisons monotones non-décroissantes
        # (les rayons sont triés croissants → les diamètres assignés doivent l'être aussi)
        best_combo = None
        best_err = float("inf")
        for combo in iter_product(denoms, repeat=len(indices_sorted)):
            diams = [DIAMETRES_MM[d] for d in combo]
            if any(diams[k] > diams[k + 1] for k in range(len(diams) - 1)):
                continue
            # Erreur = somme des |ratio_observe - ratio_theorique| sur paires consécutives
            err = 0.0
            for k in range(len(diams) - 1):
                if rayons[k] < 1 or diams[k] < 1:
                    continue
                ratio_obs = rayons[k + 1] / rayons[k]
                ratio_th = diams[k + 1] / diams[k]
                err += abs(ratio_obs - ratio_th)
            if err < best_err:
                best_err = err
                best_combo = combo

        if best_combo is None:
            continue

        # Ne ré-assigner que si l'erreur est faible (sinon ratios bizarres → mieux garder l'original)
        if best_err > 0.30:
            continue

        for k, idx in enumerate(indices_sorted):
            new_denom = best_combo[k]
            if new_denom != out[idx].denomination:
                out[idx] = ValeurPieceCombinee(
                    cercle=out[idx].cercle,
                    denomination=new_denom,
                    valeur_centimes=VALEURS_CENTIMES[new_denom],
                    confiance=out[idx].confiance,
                    groupe_couleur=groupe,
                    methode_dominante="Coherence",
                )

    return out


from modules.constants import valeur_totale as valeur_totale_combinee  # noqa: E402, F811
