from __future__ import annotations

"""Optimisation des hyperparamètres par recherche aléatoire (random search).

Chaque "epoch" teste une combinaison aléatoire de paramètres sur le dataset
annoté et enregistre les métriques obtenues. Le script affiche en fin de
course la combinaison qui a donné le meilleur F1.

Utilisation :
    # Test rapide (20 epochs, 30 images)
    python optimize.py --epochs 20 --limit 30 --seed 42

    # Optimisation complète
    python optimize.py --epochs 100 --seed 42
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import modules.segmentation as _seg_mod
import modules.classification as _det_mod
import modules.classification2 as _det2_mod

from app import rescale_annotations_to_image
from metrique import accumulate_metrics, DetectionMetrics
from modules.chargement import DatasetSample, build_dataset_index, load_sample_image
from modules.labelme_parser import load_labelme_annotation
from modules.segmentation import detect_coins


# ---------------------------------------------------------------------------
# Espace de recherche
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, tuple] = {
    # (type, low, high)  ou  ("choice", [valeurs])
    "PARAM1":         ("int",    60,   120),
    "PARAM2":         ("int",    20,    55),
    "BLUR_MEDIAN":    ("odd",     7,    21),   # entiers impairs uniquement
    "BLUR_GAUSS":     ("odd",     5,    17),   # entiers impairs uniquement
    "DP":             ("choice", [1.0, 1.1, 1.2, 1.3, 1.5]),
    "MIN_DIST_RATIO": ("float", 0.04, 0.14),
    "H_FRONTIERE":    ("float", 13.0, 21.0),
    "SIGMOID_PENTE":  ("float",  0.3,  0.8),
    "FIABILITE_K":    ("float",  4.0, 16.0),
    "INTRA_H_SIGMA":  ("float",  2.0,  7.0),
    "INTRA_S_SIGMA":  ("float", 15.0, 40.0),
}

CSV_COLUMNS = [
    "epoch", "f1", "precision", "recall",
    "PARAM1", "PARAM2", "BLUR_MEDIAN", "BLUR_GAUSS", "DP",
    "MIN_DIST_RATIO", "H_FRONTIERE", "SIGMOID_PENTE",
    "FIABILITE_K", "INTRA_H_SIGMA", "INTRA_S_SIGMA",
]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_params(rng: random.Random) -> dict[str, Any]:
    """Tire une combinaison de paramètres au hasard dans SEARCH_SPACE."""
    params: dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "int":
            params[name] = rng.randint(spec[1], spec[2])
        elif kind == "odd":
            lo = spec[1] if spec[1] % 2 == 1 else spec[1] + 1
            hi = spec[2] if spec[2] % 2 == 1 else spec[2] - 1
            params[name] = rng.choice(range(lo, hi + 1, 2))
        elif kind == "float":
            params[name] = rng.uniform(spec[1], spec[2])
        elif kind == "choice":
            params[name] = rng.choice(spec[1])
    return params


# ---------------------------------------------------------------------------
# Monkey-patching
# ---------------------------------------------------------------------------

def apply_params(params: dict[str, Any]) -> dict[str, Any]:
    """Écrase les constantes de module et retourne les valeurs originales."""
    originals: dict[str, Any] = {
        "PARAM1":         _seg_mod.PARAM1,
        "PARAM2":         _seg_mod.PARAM2,
        "BLUR_MEDIAN":    _seg_mod.BLUR_MEDIAN,
        "BLUR_GAUSS":     _seg_mod.BLUR_GAUSS,
        "DP":             _seg_mod.DP,
        "MIN_DIST_RATIO": _seg_mod.MIN_DIST_RATIO,
        "H_FRONTIERE":    _det2_mod._H_FRONTIERE,
        "SIGMOID_PENTE":  _det2_mod._SIGMOID_PENTE,
        "FIABILITE_K":    _det_mod._FIABILITE_K,
        "INTRA_H_SIGMA":  _det_mod._INTRA_H_SIGMA,
        "INTRA_S_SIGMA":  _det_mod._INTRA_S_SIGMA,
    }
    _seg_mod.PARAM1          = params["PARAM1"]
    _seg_mod.PARAM2          = params["PARAM2"]
    _seg_mod.BLUR_MEDIAN     = params["BLUR_MEDIAN"]
    _seg_mod.BLUR_GAUSS      = params["BLUR_GAUSS"]
    _seg_mod.DP              = params["DP"]
    _seg_mod.MIN_DIST_RATIO  = params["MIN_DIST_RATIO"]
    _det2_mod._H_FRONTIERE    = params["H_FRONTIERE"]
    _det2_mod._SIGMOID_PENTE  = params["SIGMOID_PENTE"]
    _det_mod._FIABILITE_K    = params["FIABILITE_K"]
    _det_mod._INTRA_H_SIGMA  = params["INTRA_H_SIGMA"]
    _det_mod._INTRA_S_SIGMA  = params["INTRA_S_SIGMA"]
    return originals


def restore_params(originals: dict[str, Any]) -> None:
    """Remet les constantes de module à leurs valeurs d'origine."""
    _seg_mod.PARAM1          = originals["PARAM1"]
    _seg_mod.PARAM2          = originals["PARAM2"]
    _seg_mod.BLUR_MEDIAN     = originals["BLUR_MEDIAN"]
    _seg_mod.BLUR_GAUSS      = originals["BLUR_GAUSS"]
    _seg_mod.DP              = originals["DP"]
    _seg_mod.MIN_DIST_RATIO  = originals["MIN_DIST_RATIO"]
    _det2_mod._H_FRONTIERE    = originals["H_FRONTIERE"]
    _det2_mod._SIGMOID_PENTE  = originals["SIGMOID_PENTE"]
    _det_mod._FIABILITE_K    = originals["FIABILITE_K"]
    _det_mod._INTRA_H_SIGMA  = originals["INTRA_H_SIGMA"]
    _det_mod._INTRA_S_SIGMA  = originals["INTRA_S_SIGMA"]


# ---------------------------------------------------------------------------
# Évaluation d'une epoch
# ---------------------------------------------------------------------------

def evaluate_epoch(samples: list[DatasetSample]) -> DetectionMetrics:
    """Évalue les métriques avec les constantes de module actuellement actives."""
    all_predictions = []
    all_annotations = []

    for sample in samples:
        try:
            image = load_sample_image(sample)
            if image is None:
                continue

            annotation = load_labelme_annotation(sample.annotation_path)
            predictions = detect_coins(image)
            ground_truth = rescale_annotations_to_image(
                annotation["circles"], annotation, image
            )

            all_predictions.append(predictions)
            all_annotations.append(ground_truth)

        except Exception as exc:
            print(f"  [warning] {sample.annotation_path.name}: {exc}")
            continue

    return accumulate_metrics(all_predictions, all_annotations)


# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------

def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Enregistre les résultats dans un CSV trié par F1 décroissant."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: r["f1"], reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_results)
    print(f"\nRésultats enregistrés dans {output_path}")


def print_best(results: list[dict[str, Any]]) -> None:
    """Affiche la meilleure combinaison de paramètres trouvée."""
    best = max(results, key=lambda r: r["f1"])
    print("\n=== Meilleurs paramètres trouvés ===")
    print(f"  Epoch     : {best['epoch']}")
    print(f"  F1        : {best['f1'] * 100:.4f}%")
    print(f"  Précision : {best['precision'] * 100:.4f}%")
    print(f"  Rappel    : {best['recall'] * 100:.4f}%")
    print()
    print("  Segmentation :")
    print(f"    PARAM1         = {best['PARAM1']}")
    print(f"    PARAM2         = {best['PARAM2']}")
    print(f"    BLUR_MEDIAN    = {best['BLUR_MEDIAN']}")
    print(f"    BLUR_GAUSS     = {best['BLUR_GAUSS']}")
    print(f"    DP             = {best['DP']}")
    print(f"    MIN_DIST_RATIO = {best['MIN_DIST_RATIO']:.4f}")
    print()
    print("  Classification couleur :")
    print(f"    H_FRONTIERE    = {best['H_FRONTIERE']:.4f}")
    print(f"    SIGMOID_PENTE  = {best['SIGMOID_PENTE']:.4f}")
    print(f"    FIABILITE_K    = {best['FIABILITE_K']:.4f}")
    print(f"    INTRA_H_SIGMA  = {best['INTRA_H_SIGMA']:.4f}")
    print(f"    INTRA_S_SIGMA  = {best['INTRA_S_SIGMA']:.4f}")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def run_optimization(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    samples, warnings = build_dataset_index(args.images_dir, args.annotations_dir)
    for w in warnings:
        print(f"[info] {w}")

    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        raise RuntimeError("Aucune image disponible. Vérifiez --images-dir et --annotations-dir.")

    n_images = len(samples)
    n_epochs = args.epochs
    width = len(str(n_epochs))

    print(f"Dataset : {n_images} image(s) | Epochs : {n_epochs} | Seed : {args.seed}")
    print(f"Baseline : F1=76.86%  P=70.33%  R=84.73%")
    print()

    results: list[dict[str, Any]] = []

    for epoch_idx in range(1, n_epochs + 1):
        params = sample_params(rng)

        originals = apply_params(params)
        try:
            metrics = evaluate_epoch(samples)
        finally:
            restore_params(originals)

        row: dict[str, Any] = {
            "epoch":          epoch_idx,
            "f1":             round(metrics.f1, 6),
            "precision":      round(metrics.precision, 6),
            "recall":         round(metrics.recall, 6),
            "PARAM1":         params["PARAM1"],
            "PARAM2":         params["PARAM2"],
            "BLUR_MEDIAN":    params["BLUR_MEDIAN"],
            "BLUR_GAUSS":     params["BLUR_GAUSS"],
            "DP":             params["DP"],
            "MIN_DIST_RATIO": round(params["MIN_DIST_RATIO"], 4),
            "H_FRONTIERE":    round(params["H_FRONTIERE"], 4),
            "SIGMOID_PENTE":  round(params["SIGMOID_PENTE"], 4),
            "FIABILITE_K":    round(params["FIABILITE_K"], 4),
            "INTRA_H_SIGMA":  round(params["INTRA_H_SIGMA"], 4),
            "INTRA_S_SIGMA":  round(params["INTRA_S_SIGMA"], 4),
        }
        results.append(row)

        print(
            f"Epoch {epoch_idx:{width}}/{n_epochs}  "
            f"F1={metrics.f1 * 100:5.2f}%  "
            f"P={metrics.precision * 100:5.2f}%  "
            f"R={metrics.recall * 100:5.2f}%  "
            f"P1={params['PARAM1']:3d}  P2={params['PARAM2']:2d}  "
            f"DP={params['DP']}  BM={params['BLUR_MEDIAN']:2d}  BG={params['BLUR_GAUSS']:2d}"
        )

    save_results(results, args.output)
    print_best(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimisation des hyperparamètres de détection par random search."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Nombre de combinaisons à tester (défaut : 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Graine aléatoire pour la reproductibilité.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Évalue sur les N premières images seulement (test rapide).",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("dataset/images"),
        help="Répertoire des images (défaut : dataset/images).",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("dataset/BDD"),
        help="Répertoire des annotations (défaut : dataset/BDD).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/optimization_results.csv"),
        help="Chemin du CSV de résultats (défaut : outputs/optimization_results.csv).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_optimization(args)
    except RuntimeError as exc:
        print(f"[erreur] {exc}", file=sys.stderr)
        sys.exit(1)
