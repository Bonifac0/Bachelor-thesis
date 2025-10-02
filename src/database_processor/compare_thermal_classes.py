import matplotlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os

matplotlib.use("Agg")

# dataset = "datasets/processed_dataset.json"
DATASET_FILE = "test.json"

if not os.path.isfile(DATASET_FILE):
    raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' does not exist.")


# {
#     "PF10417": {
#         "A0A6N7IUS6": {
#             "temp": 55.0,
#             "pred_temp": 59.0,
#             "pfam_sec": "AVQFAAATGEGVPAGWHPGQPGIKIEFDQAGTSIKPISSMIVSLLSDRTQEISWTYEVLDEETGAAYRATFIISPQSRIEYY"
#         },


def classify_temp(temp: float) -> str:
    if temp < 20:
        return "psychrophilic"
    elif 20 <= temp < 45:
        return "mesophilic"
    elif 45 <= temp < 80:
        return "thermophilic"
    else:
        return "hyperthermophilic"


# Confusion matrix labels
labels = ["psychrophilic", "mesophilic", "thermophilic", "hyperthermophilic"]
label_to_idx = {label: i for i, label in enumerate(labels)}
conf_matrix = np.zeros((4, 4), dtype=int)

# Load dataset and fill confusion matrix
with open(DATASET_FILE, "r") as f:
    data = json.load(f)
    for fam in data.values():
        for entry in fam.values():
            if "temp" in entry and "pred" in entry:
                true_class = classify_temp(entry["temp"])
                pred_class = entry["pred"]
                i = label_to_idx[true_class]
                j = label_to_idx[pred_class]
                conf_matrix[i, j] += 1

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(conf_matrix, cmap="Blues")

# Show all ticks and label them
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(labels, rotation=30)
ax.set_yticklabels(labels, rotation=30)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
plt.title("Thermal Class Confusion Matrix")

# Annotate each cell
for i in range(4):
    for j in range(4):
        ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

plt.colorbar(im)
plt.tight_layout()
plt.savefig("test_thermal_class_confusion_matrix.png")
