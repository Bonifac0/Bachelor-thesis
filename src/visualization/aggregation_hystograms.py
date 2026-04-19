import matplotlib.pyplot as plt
import numpy as np
import os
from src.training.run_model import ModelRunner
import torch


"""
python -m src.visualization.aggregation_hystograms

create hystogtam for sum function before and after transformation
create hystogram of predictor output on dataset
"""

X_PATH = "training_data/basic_1280/X.dat"
Y_PATH = "training_data/basic_1280/y.dat"
TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue


def aggregate_log_sigmoid(s, norm):
    steepnes = 4
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, 1280),
)

s = np.abs(X).sum(axis=-1)
print(f"s mean: {s.mean()}")

# sum before transformation
bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 100)
plt.hist(s, bins=bins)
plt.xscale("log")  # optional
# plt.yscale("log")  # optional
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Array Values")
plt.savefig("histogram.png", dpi=300, bbox_inches="tight")
plt.close()

# sum after transformation
norm = np.log10(np.median(s))
print(f"Norm: {norm}")
sigmoided = aggregate_log_sigmoid(s, norm)


print(sigmoided.mean())
plt.hist(sigmoided, bins=100)
plt.xlabel("Normalized Value (0-1)")
plt.ylabel("Frequency")
plt.title("Histogram (log-normalized)")
plt.savefig("histogram_normalized.png", dpi=300, bbox_inches="tight")
plt.close()

# predictor
runner = ModelRunner("2HL_64_16")
model = runner.model

# Compute embeddings
inp = torch.from_numpy(X.copy()).float().to(runner.DEVICE)

inp = (inp - runner.mean_emb) / runner.std_emb


# Forward pass
with torch.no_grad():
    logits = runner.model(inp)
    probs = torch.sigmoid(logits * 0.2)

pred_s = probs.cpu().numpy()
print(f"Median: {np.median(pred_s)}")

plt.hist(pred_s, bins=100)
plt.xlabel("Prediction")
plt.ylabel("Frequency")
plt.title("Histogram of Predictor")
plt.savefig("histogram_predictor.png", dpi=300, bbox_inches="tight")
plt.close()
