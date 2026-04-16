import numpy as np
import os

"""
python -m src.evaluations.sum_accuracy
"""


def aggregate_log_sigmoid(attribution):
    s = np.abs(attribution).sum(axis=-1)
    NORM_MEDIAN = -0.275390625
    steepnes = 4
    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - NORM_MEDIAN)))


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


s = aggregate_log_sigmoid(X)

dom_acc_count = 0
mut_acc_count = 0

for i in range(TOTAL_RESIDUES):
    if y[i] == 0 and s[i] < 0.5:
        dom_acc_count += 1
    if y[i] == 1 and s[i] > 0.5:
        mut_acc_count += 1

print(dom_acc_count / (TOTAL_RESIDUES / 2))
print(mut_acc_count / (TOTAL_RESIDUES / 2))
