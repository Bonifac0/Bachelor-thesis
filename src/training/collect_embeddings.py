from src.predictor import Classificator
from src.helpers.print_eta import ETA
from src.helpers.captum_embedding import get_captum_embedding
import json
import numpy as np
import os


"""
to run:
python -m src.training.collect_embeddings

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

    if not os.path.exists("unfinished_embeddings"):
        os.makedirs("unfinished_embeddings")

    total_residues = sum(len(prot[MODE]) for prot in protein_list)
    print(f"Total residues: {total_residues}")
    X = np.memmap(
        f"unfinished_embeddings/{MODE}_embedding.dat",
        dtype=np.float16,
        mode="w+",
        shape=(total_residues, 1280),
    )
    # y = np.memmap("y.dat", dtype=np.uint8, mode="w+", shape=(total_residues,))
    idx = 0

    protein_count = len(protein_list)
    eta = ETA(protein_count)

    print(f"Collecting embedings in mode '{MODE}'")
    print(
        f"Processing protein {0}/{protein_count} | ETA: ?",
        end="\r",
    )
    for i, entery in enumerate(protein_list):
        embedding = get_captum_embedding(mdl, entery[MODE])
        n = embedding.shape[0]

        X[idx : idx + n] = embedding
        # y[idx : idx + n] = compute_importance(entery["domain"], entery["mutant"])
        idx += n

        print(
            f"Processing protein {i + 1}/{protein_count} {eta.print_eta(i + 1)}",
            end="\r",
        )
    print()
    print("moving finnished .dat file")
    os.rename(f"unfinished_embeddings/{MODE}_embedding.dat", f"{MODE}_embedding.dat")


if __name__ == "__main__":
    classificator = Classificator()

    # switch to
    # MODE = "domain"
    MODE = "mutant"

    # INPUT_PATH = "datasets/mutants_min:13.71_hev:15.82.json"
    INPUT_PATH = "test_mutation.json"

    main(classificator)
