import numpy as np
import os

"""
python -m src.evaluations.sum_accuracy

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


# 60:40 split
split = int(0.8 * TOTAL_RESIDUES)
train_idx = indices[:split]
test_idx = indices[split:]


def aggregate_abs_log_sigmoid(attribution, norm):
    s = np.abs(attribution).sum(axis=-1)
    # norm = -0.275390625
    steepnes = 4
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


def aggregate_sigmoid(attribution, norm):
    # norm = 0.005859375
    s = attribution.sum(axis=-1)
    steepnes = 4
    return 1 / (1 + np.exp(-steepnes * (s - norm)))


def aggregate_L2(attribution, norm):
    s = np.sqrt(np.sum(attribution**2, axis=-1))
    steepnes = 4
    return 1 / (1 + np.exp(-steepnes * (s - norm)))


# --- agg abs log sigmoid ---
s_train_abs = np.abs(X[train_idx]).sum(axis=-1)
norm_abs = np.log10(np.median(s_train_abs))

print(f"Norm abs log (train only): {norm_abs}")
print(compute_acc(aggregate_abs_log_sigmoid(X[train_idx], norm_abs), y[train_idx]))

print(compute_acc(aggregate_abs_log_sigmoid(X[test_idx], norm_abs), y[test_idx]))

# --- agg sigmoid ---
s_train = X[train_idx].sum(axis=-1)
norm = np.median(s_train)


print(f"Norm sigma (train only): {norm}")
print(compute_acc(aggregate_sigmoid(X[train_idx], norm), y[train_idx]))
print(compute_acc(aggregate_sigmoid(X[test_idx], norm), y[test_idx]))

# --- agg L2 ---
norm_l2 = np.median(aggregate_L2(X[train_idx], 0))
print(norm_l2)

print(compute_acc(aggregate_L2(X[train_idx], norm_l2), y[train_idx]))
print(compute_acc(aggregate_L2(X[test_idx], norm_l2), y[test_idx]))
print((aggregate_L2(X, norm_l2) < 0.5).sum())
print((aggregate_L2(X, norm_l2) > 0.5).sum())
