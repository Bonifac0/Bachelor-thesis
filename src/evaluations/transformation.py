import numpy as np
import os

"""
python -m src.evaluations.transformation
"""


def aggregate_log_interpolation(s):
    minimum = -1.5
    maximum = 0.954
    steepnes = 10

    log_arr = np.log10(s)
    norm = (log_arr - minimum) / (maximum - minimum)
    return 1 / (1 + np.exp(-steepnes * (norm - 0.5)))


def aggregate_log_sigmoid(s, norm):
    steepnes = 4

    log_arr = np.log10(s)
    return 1 / (1 + np.exp(-steepnes * (log_arr - norm)))


X_PATH = "training_data/basic_1280/X.dat"
Y_PATH = "training_data/basic_1280/y.dat"
TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue


X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, 1280),
)

s = np.abs(X).sum(axis=-1)

norm = np.log10(np.median(s))

print(f"Norm: {norm}")

print(np.median(s))

print(np.median(aggregate_log_sigmoid(s, norm)))
print(aggregate_log_sigmoid(np.median(s), norm))
