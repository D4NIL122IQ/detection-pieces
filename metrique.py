from __future__ import annotations

from dataclasses import dataclass

from modules.labelme_parser import CircleAnnotation
from modules.segmentation import DetectedCircle


@dataclass(frozen=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0.0


def _is_match(prediction: DetectedCircle, annotation: CircleAnnotation) -> bool:
    dx = prediction.x - annotation.x
    dy = prediction.y - annotation.y
    center_distance_sq = dx * dx + dy * dy
    center_tolerance = max(10.0, annotation.radius * 0.6)
    radius_tolerance = max(8.0, annotation.radius * 0.45)
    return center_distance_sq <= center_tolerance * center_tolerance and abs(prediction.radius - annotation.radius) <= radius_tolerance


def match_circles(
    predictions: list[DetectedCircle],
    annotations: list[CircleAnnotation],
) -> tuple[int, int, int]:
    matched_predictions: set[int] = set()
    matched_annotations: set[int] = set()

    for annotation_index, annotation in enumerate(annotations):
        best_prediction_index = None
        best_gap = None

        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions:
                continue
            if not _is_match(prediction, annotation):
                continue

            gap = abs(prediction.radius - annotation.radius)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_prediction_index = prediction_index

        if best_prediction_index is not None:
            matched_predictions.add(best_prediction_index)
            matched_annotations.add(annotation_index)

    true_positives = len(matched_predictions)
    false_positives = len(predictions) - true_positives
    false_negatives = len(annotations) - len(matched_annotations)
    return true_positives, false_positives, false_negatives


def accumulate_metrics(
    all_predictions: list[list[DetectedCircle]],
    all_annotations: list[list[CircleAnnotation]],
) -> DetectionMetrics:
    tp = fp = fn = 0
    for predictions, annotations in zip(all_predictions, all_annotations, strict=False):
        image_tp, image_fp, image_fn = match_circles(predictions, annotations)
        tp += image_tp
        fp += image_fp
        fn += image_fn

    return DetectionMetrics(true_positives=tp, false_positives=fp, false_negatives=fn)
