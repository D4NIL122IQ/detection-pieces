import os
import json
import math
import shutil

IMAGE_DIR = "base_images"
JSON_DIR = "BDD"

OUTPUT_IMAGES = "dataset_yolo/images"
OUTPUT_LABELS = "dataset_yolo/labels"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

classes = {
    "1cent": 0,
    "2cents": 1,
    "5cents": 2,
    "10cents": 3,
    "20cents": 4,
    "50cents": 5,
    "1euro": 6,
    "2euros": 7
}

for json_file in os.listdir(JSON_DIR):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, json_file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = os.path.basename(data["imagePath"])
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]

        if label not in classes:
            print(f"Label inconnu ignoré : {label} dans {json_file}")
            continue

        class_id = classes[label]
        points = shape["points"]
        shape_type = shape.get("shape_type", "circle")

        if shape_type == "circle" and len(points) >= 2:
            cx, cy = points[0]
            px, py = points[1]
            radius = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

            xmin = cx - radius
            ymin = cy - radius
            xmax = cx + radius
            ymax = cy + radius
        else:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(line)

    base_name = os.path.splitext(image_name)[0]
    txt_path = os.path.join(OUTPUT_LABELS, base_name + ".txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    image_path = os.path.join(IMAGE_DIR, image_name)
    if os.path.exists(image_path):
        shutil.copy(image_path, os.path.join(OUTPUT_IMAGES, image_name))
    else:
        print(f"Image introuvable : {image_name} (référencée dans {json_file})")

print("Conversion terminée.")