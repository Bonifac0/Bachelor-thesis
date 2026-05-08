import matplotlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


matplotlib.use("Agg")

DATASET_FILE = "test_combined.json"

parser = argparse.ArgumentParser(description="Make confusion table.")
parser.add_argument("input", help="Input JSON file path")
args = parser.parse_args()
DATASET_FILE = args.input

if not os.path.isfile(DATASET_FILE):
    raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' does not exist.")


# {
#     "PF10417": {
#         "A0A6N7IUS6": {
#             "temp": 55.0,
#             "org": "Desulfotomaculum thermobenzoicum",
#             "org_id": 29376,
#             "sequence": "MTAGWWVRWLCVQCHGYKVPVTPAISPSIKPISSMIVSLLSDRTQEISWTYEVLDEETGAAYRATFIISPQSRIEYYCVYPREVGRNVDEIIRVLQAVQFAAATGEGVPAGWHPGQPGIKIEFDQAGTI",
#             "domain": "AVQFAAATGEGVPAGWHPGQPGIKIEFDQAGTSIKPISSMIVSLLSDRTQEISWTYEVLDEETGAAYRATFIISPQSRIEYY",
#             "pred_dom": [
#                 -1.804720401763916,
#                 -0.6776391267776489,
#                 2.078350067138672,
#                 -1.1388511657714844
#             ]
#         }


def classify_temp(temp: float) -> str:
    if temp < 25:
        return "psychrophilic"
    elif 25 <= temp < 45:
        return "mesophilic"
    elif 45 <= temp < 80:
        return "thermophilic"
    else:
        return "hyperthermophilic"


# Confusion matrix labels
labels = ["psychrophilic", "mesophilic", "thermophilic", "hyperthermophilic"]
label_to_idx = {label: i for i, label in enumerate(labels)}
conf_matrix = np.zeros((4, 4), dtype=int)


def array_to_class_label(arr) -> str:
    idx = np.argmax(arr)
    return labels[idx]


unclassified = 0
# Load dataset and fill confusion matrix
with open(DATASET_FILE, "r") as f:
    data = json.load(f)
    for fam in data.values():
        for entry in fam.values():
            if "temp" in entry and "pred" in entry:
                true_class = classify_temp(entry["temp"])
                pred_class = array_to_class_label(entry["pred"])
                i = label_to_idx[true_class]
                j = label_to_idx[pred_class]
                conf_matrix[i, j] += 1
            else:
                unclassified += 1

print(f"Unclassified: {unclassified}")

counts = np.sum(conf_matrix, axis=1)
total_sum = np.sum(conf_matrix)

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(conf_matrix / counts[:, np.newaxis], cmap="Blues")

# Show all ticks and label them
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(labels, rotation=30)
ax.set_yticklabels([f"{labels[i]}\n{counts[i]}" for i in range(len(labels))])
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
plt.title(f"Thermal Class Confusion Matrix for Protein (Total: {total_sum})")

# Annotate each cell
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            f"{conf_matrix[i, j] / counts[i]:.3f}",
            ha="center",
            va="center",
            color="black",
        )

plt.colorbar(im)
plt.tight_layout()
plt.savefig("class_confusion_matrix_prot.pdf")
