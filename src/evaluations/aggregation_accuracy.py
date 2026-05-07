import numpy as np
import os
from src.training.run_model import ModelRunner


"""
python -m src.evaluations.aggregation_accuracy

calculate accuracy of baseline sum method
"""


def compute_acc(data, labels):
    dom_acc_count = 0
    mut_acc_count = 0

    for i in range(data.shape[0]):
        if labels[i] == 0 and data[i] < 0.5:
            dom_acc_count += 1
        if labels[i] == 1 and data[i] > 0.5:
            mut_acc_count += 1

    return (dom_acc_count + mut_acc_count) / data.shape[0]


X_PATH = "training_data/basic_1280/X.dat"
Y_PATH = "training_data/basic_1280/y.dat"
TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue

X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, 1280),
)
y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))

# create shuffled indices
rng = np.random.default_rng(42)
indices = rng.permutation(TOTAL_RESIDUES)


split = int(0.8 * TOTAL_RESIDUES)
train_idx = indices[:split]
test_idx = indices[split:]


def aggregate_sum(attribution, norm):
    # norm = 0.005859375
    steepnes = 4
    s = attribution.sum(axis=-1)
    return 1 / (1 + np.exp(-steepnes * (s - norm)))


def aggregate_abs_sum(attribution, norm):
    # norm = -0.275390625
    steepnes = 4
    s = np.abs(attribution).sum(axis=-1)
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


def aggregate_L2(attribution, norm):
    # norm = -1.3076171875
    steepnes = 4
    s = np.sqrt(np.sum(attribution**2, axis=-1))
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


def aggregate_predictor(attribution):
    runner = ModelRunner("2HL_64_16", require_classificator=False)
    return runner.predictor_inference(attribution)


# --- agg sum ---
print("\nSum")
norm_sigm = np.median(X[train_idx].sum(axis=-1))
print(f"Norm sum (train only): {norm_sigm}")

print(compute_acc(aggregate_sum(X[train_idx], norm_sigm), y[train_idx]))
print(compute_acc(aggregate_sum(X[test_idx], norm_sigm), y[test_idx]))
# print((aggregate_sum(X, norm_sigm) < 0.5).sum())
# print((aggregate_sum(X, norm_sigm) > 0.5).sum())


# --- agg abs sum ---
print("\nAbs sum")
norm_abs = np.log10(np.median(np.abs(X[train_idx]).sum(axis=-1)))
print(f"Norm abs sum (train only): {norm_abs}")

print(compute_acc(aggregate_abs_sum(X[train_idx], norm_abs), y[train_idx]))
print(compute_acc(aggregate_abs_sum(X[test_idx], norm_abs), y[test_idx]))
# print((aggregate_abs_sum(X, norm_abs) < 0.5).sum())
# print((aggregate_abs_sum(X, norm_abs) > 0.5).sum())


# --- agg L2 ---
print("\nL2")
norm_l2 = np.log10(np.median(np.sqrt(np.sum(X[train_idx] ** 2, axis=-1))))
print(f"Norm L2: {norm_l2}")

print(compute_acc(aggregate_L2(X[train_idx], norm_l2), y[train_idx]))
print(compute_acc(aggregate_L2(X[test_idx], norm_l2), y[test_idx]))
# print((aggregate_L2(X, norm_l2) < 0.5).sum())
# print((aggregate_L2(X, norm_l2) > 0.5).sum())


# --- agg L2 ---
print("\nPredictor")
print(compute_acc(aggregate_predictor(X[train_idx]), y[train_idx]))
print(compute_acc(aggregate_predictor(X[test_idx]), y[test_idx]))
# print((aggregate_predictor(X) < 0.5).sum())
# print((aggregate_predictor(X) > 0.5).sum())
