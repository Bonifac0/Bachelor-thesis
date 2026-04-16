import numpy as np
import os

"""
python -m src.evaluations.sum_accuracy
"""


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
split = int(0.6 * TOTAL_RESIDUES)
train_idx = indices[:split]
test_idx = indices[split:]


def aggregate_log_sigmoid(attribution, norm):
    s = np.abs(attribution).sum(axis=-1)
    NORM_MEDIAN = -0.275390625
    steepnes = 4
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


def compute_acc(data, labels):
    dom_acc_count = 0
    mut_acc_count = 0

    for i in range(data.shape[0]):
        if labels[i] == 0 and data[i] < 0.5:
            dom_acc_count += 1
        if labels[i] == 1 and data[i] > 0.5:
            mut_acc_count += 1

    return (dom_acc_count + mut_acc_count) / data.shape[0]


# --- TRAIN normalization ---
s_train = np.abs(X[train_idx]).sum(axis=-1)
norm = np.log10(np.median(s_train))

print(f"Norm (train only): {norm}")
print(compute_acc(aggregate_log_sigmoid(X[train_idx], norm), y[train_idx]))

print(compute_acc(aggregate_log_sigmoid(X[test_idx], norm), y[test_idx]))
