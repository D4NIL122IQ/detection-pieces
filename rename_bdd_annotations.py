from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from modules.labelme_parser import decode_labelme_image, load_labelme_annotation


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def fingerprint(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


def build_image_index(images_dir: Path) -> dict[Path, np.ndarray]:
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in VALID_IMAGE_EXTENSIONS)
    return {path: fingerprint(cv2.imread(str(path))) for path in image_paths}


def match_annotations(
    annotations_dir: Path,
    image_index: dict[Path, np.ndarray],
) -> tuple[list[tuple[Path, Path, float]], list[Path]]:
    available = dict(image_index)
    matches: list[tuple[Path, Path, float]] = []
    unmatched: list[Path] = []

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        annotation = load_labelme_annotation(annotation_path)
        embedded = decode_labelme_image(annotation.get("image_data"))
        if embedded is None:
            unmatched.append(annotation_path)
            continue

        annotation_fp = fingerprint(embedded)
        best_image_path = None
        best_distance = None

        for image_path, image_fp in available.items():
            distance = float(np.mean((annotation_fp - image_fp) ** 2))
            if best_distance is None or distance < best_distance:
                best_image_path = image_path
                best_distance = distance

        if best_image_path is None or best_distance is None:
            unmatched.append(annotation_path)
            continue

        matches.append((annotation_path, best_image_path, best_distance))
        available.pop(best_image_path)

    return matches, unmatched


def update_annotation_file(annotation_path: Path, image_path: Path, dry_run: bool) -> Path:
    target_path = annotation_path.with_name(f"{image_path.stem}.json")
    if dry_run:
        return target_path

    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    data["imagePath"] = f"../images/{image_path.name}"

    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if target_path != annotation_path:
        annotation_path.rename(target_path)

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Renomme les annotations LabelMe pour correspondre aux images du dataset.")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("dataset/BDD"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    image_index = build_image_index(args.images_dir)
    matches, unmatched = match_annotations(args.annotations_dir, image_index)

    print(f"Images indexees      : {len(image_index)}")
    print(f"Annotations matchees : {len(matches)}")
    print(f"Annotations orphelines : {len(unmatched)}")

    for annotation_path, image_path, distance in matches[:10]:
        print(f"{annotation_path.name} -> {image_path.name} (distance={distance:.6f})")

    if args.dry_run:
        return

    renamed_targets = []
    for annotation_path, image_path, _distance in matches:
        renamed_targets.append(update_annotation_file(annotation_path, image_path, dry_run=False))

    print(f"Annotations mises a jour : {len(renamed_targets)}")
    if unmatched:
        print("Sans correspondance :")
        for annotation_path in unmatched[:20]:
            print(f"  - {annotation_path.name}")


if __name__ == "__main__":
    main()
