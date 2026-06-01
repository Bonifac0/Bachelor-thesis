import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from src.training.run_model import ModelRunner
import matplotlib.ticker as mtick

"""
python -m src.visualization.performacne_all_compare

Creates grouped bar chart with:
- Accuracy
- F1 score

for:
- Handcrafted aggregation methods
- All predictor architectures

Some predictors use different input datasets.
"""

# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------
DATASETS = {
    "basic_1280": {
        "x_path": "training_data/basic_1280/X.dat",
        "y_path": "training_data/basic_1280/y.dat",
        "features": 1280,
    },
    "basic_1280_with_len": {
        "x_path": "training_data/basic_1280_with_len/X.dat",
        "y_path": "training_data/basic_1280_with_len/y.dat",
        "features": 1281,  # 1280 + length
    },
    "all_1280*4": {
        "x_path": "training_data/all_1280*4/X.dat",
        "y_path": "training_data/all_1280*4/y.dat",
        "features": 1280 * 4,
    },
}


# --------------------------------------------------
# Load dataset helper
# --------------------------------------------------
def load_dataset(dataset_name):
    cfg = DATASETS[dataset_name]

    x_path = cfg["x_path"]
    y_path = cfg["y_path"]
    features = cfg["features"]

    total_residues = os.path.getsize(y_path)

    X = np.memmap(
        x_path,
        dtype=np.float16,
        mode="r",
        shape=(total_residues, features),
    )

    y = np.memmap(
        y_path,
        dtype=np.uint8,
        mode="r",
        shape=(total_residues,),
    )

    return X, y


# --------------------------------------------------
# Shared split indices
# --------------------------------------------------
# Assumes all datasets contain identical sample order.
basic_y_path = DATASETS["basic_1280"]["y_path"]

TOTAL_RESIDUES = os.path.getsize(basic_y_path)

rng = np.random.default_rng(42)

indices = rng.permutation(TOTAL_RESIDUES)

split = int(0.8 * TOTAL_RESIDUES)

train_idx = indices[:split]
test_idx = indices[split:]

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
datasets = {}

for dataset_name in DATASETS:
    datasets[dataset_name] = load_dataset(dataset_name)


# --------------------------------------------------
# Handcrafted aggregation methods
# --------------------------------------------------
def aggregate_sum(attribution, norm):
    steepness = 4

    s = attribution.sum(axis=-1)

    return 1 / (1 + np.exp(-steepness * (s - norm)))


def aggregate_abs_sum(attribution, norm):
    steepness = 4

    s = np.abs(attribution).sum(axis=-1)

    log_arr = np.log10(s)

    return 1 / (1 + np.exp(-steepness * (log_arr - norm)))


def aggregate_l2(attribution, norm):
    steepness = 4

    s = np.sqrt(np.sum(attribution**2, axis=-1))

    log_arr = np.log10(s)

    return 1 / (1 + np.exp(-steepness * (log_arr - norm)))


# --------------------------------------------------
# Model -> dataset mapping
# --------------------------------------------------
MODEL_DATASET = {
    "basic": "basic_1280",
    "HL_16": "basic_1280",
    "2HL_64_16": "basic_1280",
    "3HL_64_32_16": "basic_1280",
    "length": "basic_1280_with_len",
    "len_and_HL_16": "basic_1280_with_len",
    "all_class_HL_16": "all_1280*4",
}


# --------------------------------------------------
# Predictor inference
# --------------------------------------------------
def aggregate_predictor(attribution, model_name):
    runner = ModelRunner(
        model_name,
        require_classificator=False,
    )

    return runner.predictor_inference(attribution)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def evaluate(probs, labels):
    preds = (probs >= 0.5).astype(np.uint8)

    acc = accuracy_score(labels, preds)

    error_rate = 1.0 - acc

    return error_rate


# --------------------------------------------------
# Results
# --------------------------------------------------
results = {}

# --------------------------------------------------
# Handcrafted methods use basic_1280
# --------------------------------------------------
X_basic, y_basic = datasets["basic_1280"]

X_train_basic = X_basic[train_idx]
X_test_basic = X_basic[test_idx]

y_test = y_basic[test_idx]

# ---------------- SUM ----------------
print("Running SUM")
norm_sum = np.median(X_train_basic.sum(axis=-1))

probs_sum = aggregate_sum(
    X_test_basic,
    norm_sum,
)

results["sum"] = evaluate(
    probs_sum,
    y_test,
)

# ---------------- ABS SUM ----------------
print("Running ABS SUM")
norm_abs = np.log10(np.median(np.abs(X_train_basic).sum(axis=-1)))

probs_abs = aggregate_abs_sum(
    X_test_basic,
    norm_abs,
)

results["abs_sum"] = evaluate(
    probs_abs,
    y_test,
)

# ---------------- L2 ----------------
print("Running L2")
norm_l2 = np.log10(np.median(np.sqrt(np.sum(X_train_basic**2, axis=-1))))

probs_l2 = aggregate_l2(
    X_test_basic,
    norm_l2,
)

results["L2"] = evaluate(
    probs_l2,
    y_test,
)

# --------------------------------------------------
# Run all predictor models
# --------------------------------------------------
for model_name, dataset_name in MODEL_DATASET.items():
    print(f"Running model: {model_name}")
    X_current, y_current = datasets[dataset_name]

    X_test = X_current[test_idx]
    y_test = y_current[test_idx]

    probs = aggregate_predictor(
        X_test,
        model_name,
    )

    results[model_name] = evaluate(
        probs,
        y_test,
    )

# --------------------------------------------------
# Print summary
# --------------------------------------------------
print("\n====================================")
print("Performance Summary")
print("====================================\n")

for name, error_rate in results.items():
    print(f"{name:22s} | Error Rate: {error_rate:.4f}")

# --------------------------------------------------
# Plot
# --------------------------------------------------
methods = list(results.keys())

error_rates = [results[m] for m in methods]

x = np.arange(len(methods))

colors = ["tab:blue"] * 3 + ["tab:orange"] * (len(methods) - 3)
fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(
    x,
    error_rates,
    width=0.8,
    color=colors,
)
# Add values above bars
for bar, value in zip(bars, error_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=18,
    )

# --------------------------------------------------
# Formatting
# --------------------------------------------------
ax.set_title("Error Rate Comparison of Aggregation Technologies", fontsize=24)
ax.set_ylabel("Error Rate", fontsize=18)
ax.set_xlabel("Predictor type", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=14)
plt.ylim(0, 0.4)

ax.set_xticks(x)

ax.set_xticklabels(
    methods,
    rotation=25,
    ha="right",
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

plt.tight_layout()

plt.savefig(
    "error_rate_comparison.pdf",
    format="pdf",
    bbox_inches="tight",
)

plt.close()
