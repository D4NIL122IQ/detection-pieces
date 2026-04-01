from __future__ import annotations

"""Evaluation de la classification des valeurs de pieces sur le dataset.

Pipeline : chargement image → detection cercles → classification couleur/taille
→ comparaison avec la verite terrain LabelMe → metriques F1/precision/rappel.
"""

import argparse
from pathlib import Path

import cv2

from app import rescale_annotations_to_image
from metriqueVT import compute_valeur_metrics, print_valeur_metrics
from modules.chargement import build_dataset_index, load_sample_image
from modules.determination import classify_by_color_and_size
from modules.labelme_parser import load_labelme_annotation
from modules.segmentation import detect_coins


def evaluate(
    images_dir: Path,
    annotations_dir: Path,
    limit: int | None = None,
) -> None:
    samples, warnings = build_dataset_index(images_dir, annotations_dir)
    if limit is not None:
        samples = samples[:limit]

    if not samples:
        print("Aucune paire image/annotation trouvee.")
        return

    for w in warnings:
        print(f"[warning] {w}")

    all_predictions = []
    all_annotations = []
    skipped = 0

    for sample in samples:
        image = load_sample_image(sample)
        if image is None:
            print(f"[warning] Image illisible : {sample.annotation_path}")
            skipped += 1
            continue

        annotation = load_labelme_annotation(sample.annotation_path)
        gt_circles = rescale_annotations_to_image(
            annotation["circles"], annotation, image,
        )

        detected = detect_coins(image)
        valuations = classify_by_color_and_size(detected, image)

        all_predictions.append(valuations)
        all_annotations.append(gt_circles)

    print(f"\nImages evaluees : {len(all_predictions)} (ignorees : {skipped})")

    metrics = compute_valeur_metrics(all_predictions, all_annotations)
    print_valeur_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation de la classification des valeurs de pieces.",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=Path("dataset/images"),
        help="Dossier des images (default: dataset/images)",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=Path("dataset/BDD"),
        help="Dossier des annotations LabelMe (default: dataset/BDD)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Nombre max d'images a evaluer",
    )
    args = parser.parse_args()
    evaluate(args.images_dir, args.annotations_dir, args.limit)


if __name__ == "__main__":
    main()
