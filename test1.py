import os
from collections import defaultdict
from pathlib import Path

img_dir = Path("dataset_yolo/images")
image_exts = {".png", ".jpg", ".jpeg", ".webp"}

by_stem = defaultdict(list)

for f in img_dir.iterdir():
    if f.is_file() and f.suffix.lower() in image_exts:
        by_stem[f.stem].append(f.name)

duplicates = {stem: files for stem, files in by_stem.items() if len(files) > 1}

print("Nombre de stems dupliqués :", len(duplicates))
for stem, files in sorted(duplicates.items()):
    print(stem, "->", files)