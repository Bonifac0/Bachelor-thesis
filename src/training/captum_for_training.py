from src.predictor import Classificator
from src.heplers.print_eta import ETA
from src.heplers.captum_embedding import get_captum_embedding
import json
import numpy as np


"""
to run:
python -m src.training.captum_for_training

because pyhon need to load packages
"""


def compute_importance(baseline: str, mutant: str) -> np.ndarray:
    assert len(baseline) == len(mutant)
    return np.fromiter(
        (b != m for b, m in zip(baseline, mutant)), dtype=np.uint8, count=len(baseline)
    )


def main(mdl: Classificator):
    with open(INPUT_PATH, "r") as f:
        protein_list = json.load(f)

    total_residues = sum(len(prot["domain"]) for prot in protein_list)
    print(f"Total residues: {total_residues}")
    X = np.memmap("X.dat", dtype=np.float16, mode="w+", shape=(total_residues, 1280))
    y = np.memmap("y.dat", dtype=np.uint8, mode="w+", shape=(total_residues,))
    idx = 0

    protein_count = len(protein_list)
    eta = ETA(protein_count)

    print()
    print(
        f"Processing protein {0}/{protein_count} | ETA: ?",
        end="\r",
    )
    for i, entery in enumerate(protein_list):
        embedding = get_captum_embedding(mdl, entery["domain"])
        n = embedding.shape[0]
        # print(n)
        # print(len(entery["domain"]))

        X[idx : idx + n] = embedding
        y[idx : idx + n] = compute_importance(entery["domain"], entery["mutant"])
        idx += n

        print(
            f"Processing protein {i + 1}/{protein_count} {eta.print_eta(i + 1)}",
            end="\r",
        )
    print()

    print(X.shape)
    print(y.shape)


if __name__ == "__main__":
    classificator = Classificator()

    INPUT_PATH = "test_mutation.json"

    main(classificator)
