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


def aggregate_log_sigmoid(attribution, norm):
    # s = np.abs(attribution).sum(axis=-1)
    s = attribution.sum(axis=-1)
    NORM_MEDIAN = -0.275390625
    steepnes = 18
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, 1280),
)


#########################
# without abs
#########################

s = X.sum(axis=-1)
med = np.median(s)
print(f"Default median: {med}")
print(f"Min: {np.min(s)}")
print(f"Max: {np.max(s)}")


#####
plt.hist(s, bins=100)
plt.xlabel("Normalized Value (0-1)")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Histogram (log-normalized)")
plt.savefig("s.pdf", bbox_inches="tight")
plt.close()
#####

#########################
# before normalization
#########################

bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 100)

plt.hist(s, bins=bins)

# vertical line at median
plt.axvline(
    med,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Median = {med:.2e}",
)

plt.xscale("log")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Array Values")
plt.legend()

plt.savefig("histogram.pdf", bbox_inches="tight")
plt.close()

#########################
# normalized
#########################

norm = np.log10(np.median(s))
print(f"Norm: {norm}")
sigmoided = aggregate_log_sigmoid(X, norm)

print(np.median(sigmoided))
plt.hist(sigmoided, bins=100)
plt.xlabel("Normalized Value (0-1)")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Histogram (log-normalized)")
plt.savefig("histogram_normalized.pdf", bbox_inches="tight")
plt.close()

#########################
# predictor
#########################

runner = ModelRunner("2HL_64_16")
model = runner.model

# Compute attributions
inp = torch.from_numpy(X.copy()).float().to(runner.DEVICE)

inp = (inp - runner.mean_atr) / runner.std_atr

# Forward pass
with torch.no_grad():
    logits = runner.model(inp)
    probs = torch.sigmoid(logits)

pred_s = probs.cpu().numpy()
print(f"Median: {np.median(pred_s)}")

plt.hist(pred_s, bins=100)
plt.xlabel("Prediction")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Histogram of Predictor")
plt.savefig("histogram_predictor.pdf", bbox_inches="tight")
plt.close()
