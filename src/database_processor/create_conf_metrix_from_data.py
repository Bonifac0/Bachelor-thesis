import numpy as np
import matplotlib.pyplot as plt

# Labels
labels = [
    "psychrophilic",
    "mesophilic",
    "thermophilic",
    "hyperthermophilic",
]

# Extracted row counts from image
counts = np.array([2564, 153824, 30143, 1582])

# Extracted normalized confusion matrix values from image
norm_conf_matrix = np.array(
    [
        [0.733, 0.125, 0.136, 0.006],
        [0.436, 0.352, 0.206, 0.005],
        [0.095, 0.090, 0.738, 0.077],
        [0.010, 0.002, 0.138, 0.850],
    ]
)

# Reconstruct approximate raw confusion matrix
conf_matrix = np.round(norm_conf_matrix * counts[:, np.newaxis]).astype(int)

# Ensure row sums exactly match original counts
for i in range(len(counts)):
    diff = counts[i] - np.sum(conf_matrix[i])
    conf_matrix[i, np.argmax(conf_matrix[i])] += diff

# Total samples
total_sum = np.sum(conf_matrix)

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4.8))

im = ax.imshow(
    conf_matrix / counts[:, np.newaxis],
    cmap="Blues",
    vmin=0,
    vmax=np.max(norm_conf_matrix),
)

# Ticks and labels
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))

ax.set_xticklabels(labels, rotation=30)
ax.set_yticklabels([f"{labels[i]}\n{counts[i]}" for i in range(len(labels))])

ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")

plt.title(f"Thermal Class Confusion Matrix for Domain (Total: {total_sum})")

# Annotate cells
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

# Colorbar
plt.colorbar(im)

plt.tight_layout()

# Save as PDF
plt.savefig("class_confusion_matrix_domain.pdf", format="pdf")
