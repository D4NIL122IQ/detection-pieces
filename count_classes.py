import os
from collections import Counter

labels_dir = "dataset_yolo/labels"

counter = Counter()

for root, _, files in os.walk(labels_dir):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

classes = {
    0: "1cent",
    1: "2cents",
    2: "5cents",
    3: "10cents",
    4: "20cents",
    5: "50cents",
    6: "1euro",
    7: "2euros"
}

print("Nombre d'instances par classe :\n")
for k, v in counter.items():
    print(f"{classes[k]} : {v}")