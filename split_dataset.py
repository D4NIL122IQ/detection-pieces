import os
import random
import shutil

# Dossiers source
images_dir = "dataset_yolo/images"
labels_dir = "dataset_yolo/labels"

# Dossiers destination
output_images_train = "dataset_yolo/images/train"
output_images_val = "dataset_yolo/images/val"
output_images_test = "dataset_yolo/images/test"

output_labels_train = "dataset_yolo/labels/train"
output_labels_val = "dataset_yolo/labels/val"
output_labels_test = "dataset_yolo/labels/test"

# Création des dossiers
for folder in [
    output_images_train, output_images_val, output_images_test,
    output_labels_train, output_labels_val, output_labels_test
]:
    os.makedirs(folder, exist_ok=True)

# Extensions d’images acceptées
image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Récupérer les images
images = [f for f in os.listdir(images_dir) if f.endswith(image_extensions)]

# Garder seulement celles qui ont un label correspondant
valid_images = []
for img in images:
    base_name = os.path.splitext(img)[0]
    label_path = os.path.join(labels_dir, base_name + ".txt")
    if os.path.exists(label_path):
        valid_images.append(img)
    else:
        print(f"Pas de label pour : {img}")

print(f"Nombre d'images valides : {len(valid_images)}")

# Mélanger aléatoirement
random.seed(42)
random.shuffle(valid_images)

# Répartition 70 / 20 / 10
n = len(valid_images)
train_count = int(n * 0.7)
val_count = int(n * 0.2)
test_count = n - train_count - val_count

train_images = valid_images[:train_count]
val_images = valid_images[train_count:train_count + val_count]
test_images = valid_images[train_count + val_count:]

print(f"Train : {len(train_images)}")
print(f"Val   : {len(val_images)}")
print(f"Test  : {len(test_images)}")

def copy_files(image_list, out_img_dir, out_lbl_dir):
    for img in image_list:
        base_name = os.path.splitext(img)[0]
        label_file = base_name + ".txt"

        src_img = os.path.join(images_dir, img)
        src_lbl = os.path.join(labels_dir, label_file)

        dst_img = os.path.join(out_img_dir, img)
        dst_lbl = os.path.join(out_lbl_dir, label_file)

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

copy_files(train_images, output_images_train, output_labels_train)
copy_files(val_images, output_images_val, output_labels_val)
copy_files(test_images, output_images_test, output_labels_test)

print("Découpage terminé.")