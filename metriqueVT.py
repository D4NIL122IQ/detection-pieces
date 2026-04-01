from __future__ import annotations

"""Metriques de classification des valeurs de pieces.

Compare la denomination predite par les algorithmes de determination
a la verite terrain issue des annotations LabelMe.
Calcule precision, rappel et F1 par classe et globalement (micro/macro).
"""

from dataclasses import dataclass, field

from modules.determination import ValeurPiece
from modules.labelme_parser import CircleAnnotation
from modules.segmentation import DetectedCircle

# Reutilise le matching spatial de metrique.py
from metrique import _is_match


DENOMINATIONS = ["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]

# Mapping des labels LabelMe (verite terrain) vers les denominations internes
_LABEL_TO_DENOM: dict[str, str] = {
    "1cent":   "1c",
    "2cent":   "2c",
    "2cents":  "2c",
    "5cent":   "5c",
    "5cents":  "5c",
    "10cent":  "10c",
    "10cents": "10c",
    "20cent":  "20c",
    "20cents": "20c",
    "50cent":  "50c",
    "50cents": "50c",
    "1euro":   "1e",
    "2euro":   "2e",
    "2euros":  "2e",
}


def normalize_label(label: str) -> str:
    """Convertit un label LabelMe en denomination interne (ex: '2euros' → '2e')."""
    return _LABEL_TO_DENOM.get(label.strip().lower(), label)


@dataclass(frozen=True)
class ClassMetrics:
    """Precision / Rappel / F1 pour une classe donnee."""

    denomination: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        d = p + r
        return 2 * p * r / d if d else 0.0


@dataclass(frozen=True)
class ValeurMetrics:
    """Metriques globales et par classe pour la classification des valeurs."""

    par_classe: dict[str, ClassMetrics] = field(default_factory=dict)
    total_matched: int = 0
    total_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.total_correct / self.total_matched if self.total_matched else 0.0

    @property
    def micro_precision(self) -> float:
        tp = sum(m.tp for m in self.par_classe.values())
        fp = sum(m.fp for m in self.par_classe.values())
        d = tp + fp
        return tp / d if d else 0.0

    @property
    def micro_recall(self) -> float:
        tp = sum(m.tp for m in self.par_classe.values())
        fn = sum(m.fn for m in self.par_classe.values())
        d = tp + fn
        return tp / d if d else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        d = p + r
        return 2 * p * r / d if d else 0.0

    @property
    def macro_precision(self) -> float:
        actives = [m for m in self.par_classe.values() if m.tp + m.fp + m.fn > 0]
        return sum(m.precision for m in actives) / len(actives) if actives else 0.0

    @property
    def macro_recall(self) -> float:
        actives = [m for m in self.par_classe.values() if m.tp + m.fp + m.fn > 0]
        return sum(m.recall for m in actives) / len(actives) if actives else 0.0

    @property
    def macro_f1(self) -> float:
        actives = [m for m in self.par_classe.values() if m.tp + m.fp + m.fn > 0]
        return sum(m.f1 for m in actives) / len(actives) if actives else 0.0


def match_predictions_to_ground_truth(
    predictions: list[ValeurPiece],
    annotations: list[CircleAnnotation],
) -> list[tuple[ValeurPiece, CircleAnnotation]]:
    """Associe chaque prediction a l'annotation la plus proche (matching spatial).

    Retourne les paires (prediction, annotation) matchees.
    """
    matched_pred: set[int] = set()
    matched_anno: set[int] = set()
    pairs: list[tuple[ValeurPiece, CircleAnnotation]] = []

    for a_idx, anno in enumerate(annotations):
        best_p_idx: int | None = None
        best_gap: float | None = None

        for p_idx, pred in enumerate(predictions):
            if p_idx in matched_pred:
                continue
            if not _is_match(pred.cercle, anno):
                continue
            gap = abs(pred.cercle.radius - anno.radius)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_p_idx = p_idx

        if best_p_idx is not None:
            matched_pred.add(best_p_idx)
            matched_anno.add(a_idx)
            pairs.append((predictions[best_p_idx], anno))

    return pairs


def compute_valeur_metrics(
    all_predictions: list[list[ValeurPiece]],
    all_annotations: list[list[CircleAnnotation]],
) -> ValeurMetrics:
    """Calcule les metriques de classification sur tout le dataset.

    Pour chaque image, les predictions sont matchees spatialement aux
    annotations. Pour chaque paire matchee, on compare la denomination
    predite au label de verite terrain.

    Args:
        all_predictions: liste de predictions par image (sortie de classify_*).
        all_annotations: liste d'annotations par image (depuis LabelMe).
    """
    # Compteurs par classe : {denom: [tp, fp, fn]}
    counts: dict[str, list[int]] = {d: [0, 0, 0] for d in DENOMINATIONS}
    total_matched = 0
    total_correct = 0

    for preds, annos in zip(all_predictions, all_annotations, strict=False):
        pairs = match_predictions_to_ground_truth(preds, annos)
        matched_pred_indices: set[int] = set()
        matched_anno_labels: set[int] = set()

        for pred, anno in pairs:
            total_matched += 1
            pred_denom = pred.denomination
            gt_denom = normalize_label(anno.label)

            if pred_denom == gt_denom:
                total_correct += 1
                if pred_denom in counts:
                    counts[pred_denom][0] += 1  # TP
            else:
                # FP pour la classe predite (predit X mais c'est Y)
                if pred_denom in counts:
                    counts[pred_denom][1] += 1
                # FN pour la classe reelle (c'est Y mais predit X)
                if gt_denom in counts:
                    counts[gt_denom][2] += 1

        # Predictions non matchees → FP pour leur classe
        for p_idx, pred in enumerate(preds):
            already_matched = any(
                pred is pair_pred for pair_pred, _ in pairs
            )
            if not already_matched and pred.denomination in counts:
                counts[pred.denomination][1] += 1

        # Annotations non matchees → FN pour leur classe
        matched_anno_set = {id(anno) for _, anno in pairs}
        for anno in annos:
            gt = normalize_label(anno.label)
            if id(anno) not in matched_anno_set and gt in counts:
                counts[gt][2] += 1

    par_classe = {
        denom: ClassMetrics(denomination=denom, tp=c[0], fp=c[1], fn=c[2])
        for denom, c in counts.items()
    }

    return ValeurMetrics(
        par_classe=par_classe,
        total_matched=total_matched,
        total_correct=total_correct,
    )


def print_valeur_metrics(metrics: ValeurMetrics) -> None:
    """Affiche les metriques de classification dans la console."""
    print("\n=== Metriques de classification des valeurs ===\n")
    print(f"{'Classe':<8} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rappel':>7} {'F1':>7}")
    print("-" * 50)

    for denom in DENOMINATIONS:
        m = metrics.par_classe.get(denom)
        if m is None:
            continue
        if m.tp + m.fp + m.fn == 0:
            continue
        print(
            f"{denom:<8} {m.tp:>4} {m.fp:>4} {m.fn:>4}"
            f" {m.precision:>6.1%} {m.recall:>6.1%} {m.f1:>6.1%}"
        )

    total_tp = sum(m.tp for m in metrics.par_classe.values())
    total_fp = sum(m.fp for m in metrics.par_classe.values())
    total_fn = sum(m.fn for m in metrics.par_classe.values())
    print(
        f"{'TOTAL':<8} {total_tp:>4} {total_fp:>4} {total_fn:>4}"
        f" {metrics.micro_precision:>6.1%} {metrics.micro_recall:>6.1%} {metrics.micro_f1:>6.1%}"
    )
    print("-" * 50)

    total_gt = total_tp + total_fn
    detectees = total_tp + total_fp
    non_detectees = total_fn
    print(f"Pieces dans la verite terrain : {total_gt}")
    print(f"Pieces detectees             : {detectees}")
    print(f"Pieces correctement classees : {total_tp}")
    print(f"Pieces non detectees         : {non_detectees}")
    print()
    print(f"Accuracy globale : {metrics.accuracy:.1%} ({metrics.total_correct}/{metrics.total_matched})")
    print(f"Micro  — Prec: {metrics.micro_precision:.1%}  Rappel: {metrics.micro_recall:.1%}  F1: {metrics.micro_f1:.1%}")
    print(f"Macro  — Prec: {metrics.macro_precision:.1%}  Rappel: {metrics.macro_recall:.1%}  F1: {metrics.macro_f1:.1%}")
