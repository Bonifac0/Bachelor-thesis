from captum.attr import IntegratedGradients
import torch
import numpy as np
import os
from src.training.model_definitions import ImportancePredictorWithLength

"""
python -m src.evaluations.captum_on_predictor

obsolete, so far no useful results
apply captum on predikcor to see importance of input
"""

GET_FIRST = 1

MODE = "basic_1280_with_len"
X_PATH = f"training_data/{MODE}/X.dat"
Y_PATH = f"training_data/{MODE}/y.dat"
LENGTHS_PATH = f"training_data/{MODE}/lengths.dat"
AA_PATH = f"training_data/{MODE}/amino_acids.txt"

TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue
X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, ImportancePredictorWithLength.FEATURES),
)
# y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))

inp = torch.tensor(X[:GET_FIRST, :], dtype=torch.float32)
inp.requires_grad_(True)

# baseline = torch.tensor(y[:GET_FIRST])
baseline = torch.zeros_like(inp)
baseline[:, :1280] = 0  # embeddings off
baseline[:, 1280] = 0  # average length

model = ImportancePredictorWithLength()
model.eval()

ig = IntegratedGradients(model)

attributions, delta = ig.attribute(
    inputs=inp,
    baselines=baseline,
    return_convergence_delta=True,
)

embedding_attr = attributions[:, :1280]  # per-dimension importance
length_attr = attributions[:, 1280]  # importance of length feature

print(f"Delta: {delta}")
print(f"Len importance: {length_attr}")
print(sum(max(embedding_attr)))
