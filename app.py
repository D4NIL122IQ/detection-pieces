from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from metrique import accumulate_metrics
from modules.chargement import build_dataset_index, load_sample_image
from modules.labelme_parser import load_labelme_annotation
from modules.segmentation import detect_coins, draw_circles


def run_single_image(image_path: Path, output_path: Path | None = None) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {image_path}")

    predictions = detect_coins(image)
    print(f"{image_path}: {len(predictions)} piece(s) detectee(s)")
    for index, circle in enumerate(predictions, start=1):
        print(f"  {index:02d}. x={circle.x}, y={circle.y}, r={circle.radius}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visual = draw_circles(image, predictions)
        cv2.imwrite(str(output_path), visual)
        print(f"Visualisation enregistree dans {output_path}")


def evaluate_dataset(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> None:
    samples, warnings = build_dataset_index(images_dir, annotations_dir)
    if limit is not None:
        samples = samples[:limit]

    if not samples:
        raise RuntimeError("Aucune paire image/annotation disponible pour l'evaluation.")

    for warning in warnings:
        print(f"[warning] {warning}")

    all_predictions = []
    all_annotations = []

    for sample in samples:
        image = load_sample_image(sample)
        if image is None:
            print(f"[warning] Image illisible ignoree: {sample.annotation_path}")
            continue

        annotation = load_labelme_annotation(sample.annotation_path)
        predictions = detect_coins(image)
        ground_truth = annotation["circles"]

        all_predictions.append(predictions)
        all_annotations.append(ground_truth)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            visual = draw_circles(
                image,
                predictions,
                ground_truth=[(circle.x, circle.y, circle.radius) for circle in ground_truth],
            )
            target_name = sample.image_path.name if sample.image_path is not None else f"{sample.annotation_path.stem}.jpg"
            cv2.imwrite(str(output_dir / target_name), visual)

    metrics = accumulate_metrics(all_predictions, all_annotations)
    print(f"Images evaluees : {len(all_predictions)}")
    print(f"Precision      : {metrics.precision * 100:.2f}%")
    print(f"Rappel         : {metrics.recall * 100:.2f}%")
    print(f"F1             : {metrics.f1 * 100:.2f}%")
    print(f"TP / FP / FN   : {metrics.true_positives} / {metrics.false_positives} / {metrics.false_negatives}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detection de pieces d'euros par transformee de Hough.")
    parser.add_argument("--image", type=Path, help="Chemin vers une image unique a traiter.")
    parser.add_argument("--output", type=Path, help="Chemin de sortie pour une visualisation sur image unique.")
    parser.add_argument("--evaluate", action="store_true", help="Evalue la detection sur le dataset.")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("dataset/BDD"))
    parser.add_argument("--vis-dir", type=Path, help="Dossier de sortie pour les visualisations dataset.")
    parser.add_argument("--limit", type=int, help="Nombre maximal d'images a evaluer.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.image is not None:
        run_single_image(args.image, args.output)
        return

    if args.evaluate:
        evaluate_dataset(args.images_dir, args.annotations_dir, args.vis_dir, args.limit)
        return

    parser.error("Utilise --image <chemin> ou --evaluate.")


if __name__ == "__main__":
    main()
