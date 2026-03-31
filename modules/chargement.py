from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from modules.labelme_parser import decode_labelme_image, load_labelme_annotation


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path | None
    annotation_path: Path


def _normalized_stem(path: Path) -> str:
    return path.stem.strip().casefold()


def build_dataset_index(
    images_dir: str | Path = "dataset/images",
    annotations_dir: str | Path = "dataset/BDD",
) -> tuple[list[DatasetSample], list[str]]:
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    annotation_paths = sorted(
        path for path in annotations_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json"
    )

    warnings: list[str] = []
    by_stem = {_normalized_stem(path): path for path in image_paths}

    samples: list[DatasetSample] = []
    matched_file_count = 0
    embedded_image_count = 0

    for annotation_path in annotation_paths:
        annotation = load_labelme_annotation(annotation_path)
        image_path = None

        image_name = annotation.get("image_path")
        if image_name:
            candidate = by_stem.get(_normalized_stem(Path(image_name)))
            if candidate is not None:
                image_path = candidate
                matched_file_count += 1
            else:
                embedded_image_count += 1
        else:
            embedded_image_count += 1

        samples.append(DatasetSample(image_path=image_path, annotation_path=annotation_path))

    if matched_file_count == 0:
        warnings.append(
            "Aucune annotation ne pointe vers un fichier de `dataset/images`. "
            "L'evaluation utilisera les images embarquees dans les JSON LabelMe."
        )
    elif embedded_image_count:
        warnings.append(
            f"{embedded_image_count} annotation(s) sans image correspondante dans `dataset/images`; "
            "repli sur l'image embarquee pour celles-ci."
        )

    missing_annotations = max(0, len(image_paths) - len(annotation_paths))
    if missing_annotations:
        warnings.append(
            f"{missing_annotations} image(s) de `dataset/images` n'ont pas d'annotation exploitable."
        )

    return samples, warnings


def inspect_sample(sample: DatasetSample) -> dict:
    annotation = load_labelme_annotation(sample.annotation_path)
    return {
        "image_path": sample.image_path,
        "annotation_path": sample.annotation_path,
        "annotation": annotation,
    }


def load_sample_image(sample: DatasetSample) -> np.ndarray | None:
    if sample.image_path is not None:
        image = cv2.imread(str(sample.image_path))
        if image is not None:
            return image

    annotation = load_labelme_annotation(sample.annotation_path)
    return decode_labelme_image(annotation.get("image_data"))
