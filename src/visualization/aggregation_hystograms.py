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


def aggregate_sum(attribution):
    norm = 0.005859375
    steepnes = 20
    s = attribution.sum(axis=-1)
    return 1 / (1 + np.exp(-steepnes * (s - norm)))


def aggregate_abs_sum(attribution):
    norm = -0.275390625
    steepnes = 20
    s = np.abs(attribution).sum(axis=-1)
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


def aggregate_L2(attribution):
    norm = -1.3076171875
    steepnes = 12
    s = np.sqrt(np.sum(attribution**2, axis=-1))
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


def plot_default(data):
    s = data.sum(axis=-1)
    med = np.median(s)
    print(f"Default median: {med}")
    print(f"Min: {np.min(s)}")
    print(f"Max: {np.max(s)}")

    plt.hist(s, bins=100)
    plt.xlabel("Normalized Value (0-1)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Histogram (log-normalized)")
    plt.savefig("sum_agr_hist/default.pdf", bbox_inches="tight")
    plt.close()


def log_plot(data):  # obsolete
    s = data.sum(axis=-1)
    med = np.median(s)

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
    plt.savefig("sum_agr_hist/default.pdf", bbox_inches="tight")
    plt.close()


def plot_sum(data):
    plt.hist(aggregate_sum(data), bins=100)
    plt.xlabel("Normalized Value (0-1)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Histogram sum aggregation")
    plt.savefig("sum_agr_hist/sum.pdf", bbox_inches="tight")
    plt.close()


def plot_abs_sum(data):
    plt.hist(aggregate_abs_sum(data), bins=100)
    plt.xlabel("Normalized Value (0-1)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Histogram log10 abs sum aggregation")
    plt.savefig("sum_agr_hist/abs_sum.pdf", bbox_inches="tight")
    plt.close()


def plot_L2(data):
    plt.hist(aggregate_L2(data), bins=100)
    plt.xlabel("Normalized Value (0-1)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Histogram log10 L2 aggregation")
    plt.savefig("sum_agr_hist/L2.pdf", bbox_inches="tight")
    plt.close()


def plot_predictor(data):
    runner = ModelRunner("2HL_64_16", require_classificator=False)

    # Compute attributions
    inp = torch.from_numpy(data.copy()).float().to(runner.DEVICE)

    inp = (inp - runner.mean_atr) / runner.std_atr

    # Forward pass
    with torch.no_grad():
        logits = runner.model(inp)
        probs = torch.sigmoid(logits)

    pred_s = probs.cpu().numpy()

    plt.hist(pred_s, bins=100)
    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Histogram of Predictor")
    plt.savefig("sum_agr_hist/predictor.pdf", bbox_inches="tight")
    plt.close()


def plot_comparison(data):
    runner = ModelRunner("2HL_64_16", require_classificator=False)

    # --- Method 1: abs sum aggregation ---
    abs_sum_values = aggregate_abs_sum(data)

    # --- Method 2: predictor output ---
    inp = torch.from_numpy(data.copy()).float().to(runner.DEVICE)
    inp = (inp - runner.mean_atr) / runner.std_atr

    with torch.no_grad():
        logits = runner.model(inp)
        probs = torch.sigmoid(logits)

    pred_s = probs.cpu().numpy().flatten()

    # --- Combined histogram ---
    plt.figure(figsize=(8, 5))

    plt.hist(
        abs_sum_values,
        bins=100,
        alpha=0.5,
        label="Abs Sum Aggregation",
    )

    plt.hist(
        pred_s,
        bins=100,
        alpha=0.5,
        label="Predictor",
    )

    plt.axvline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Threshold = 0.5",
    )

    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Comparison of Aggregation vs Predictor")
    plt.legend()

    plt.savefig("graphs/agr_vs_pred_comparison.pdf", bbox_inches="tight")
    plt.close()


X_PATH = "training_data/basic_1280/X.dat"
Y_PATH = "training_data/basic_1280/y.dat"
TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue
X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, 1280),
)

# print("\nploting default")
# plot_default(X)

# print("\nploting sum")
# plot_sum(X)

# print("\nploting abs sum")
# plot_abs_sum(X)

# print("\nploting L2")
# plot_L2(X)

# print("\nploting predictor")
# plot_predictor(X)

plot_comparison(X)
