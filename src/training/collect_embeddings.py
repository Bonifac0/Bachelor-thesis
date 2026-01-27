from src.predictor import Classificator
from src.helpers.print_eta import ETA
from src.helpers.captum_embedding import get_captum_embedding
import json
import numpy as np
import os
import argparse


"""
to run:
python -m src.training.collect_embeddings --mode domain
python -m src.training.collect_embeddings --mode mutant

for testing
python -m src.training.collect_embeddings --mode domain --input-path test_mutation.json
"""

FEATURES = 1280 * 4


def collect_embeddings(mdl: Classificator, mode: str, input_path: str):
    """
    Collect embeddings for proteins using captum
    Two possible modes of collections: domain, mutant
    Saves results to memmap (unfinished to its folder)
    """
    with open(input_path, "r") as f:
        protein_list = json.load(f)

    if not os.path.exists("unfinished_embeddings"):
        os.makedirs("unfinished_embeddings")

    total_residues = sum(len(prot[mode]) for prot in protein_list)
    print(f"Total residues: {total_residues}")

    X = np.memmap(
        f"unfinished_embeddings/{mode}_embedding.dat",
        dtype=np.float16,
        mode="w+",
        shape=(total_residues, FEATURES),
    )

    idx = 0
    protein_count = len(protein_list)
    eta = ETA(protein_count)

    print(f"Collecting embeddings in mode '{mode}'")
    print(
        f"Processing protein {0}/{protein_count} | ETA: ?",
        end="\r",
    )

    for i, entry in enumerate(protein_list):
        embedding = get_captum_embedding(mdl, entry[mode])
        n = embedding.shape[0]

        X[idx : idx + n] = embedding
        idx += n

        print(
            f"Processing protein {i + 1}/{protein_count} {eta.print_eta(i + 1)}",
            end="\r",
        )

    print()
    print("moving finished .dat file")
    os.rename(
        f"unfinished_embeddings/{mode}_embedding.dat",
        f"{mode}_embedding.dat",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["domain", "mutant"],
        help="Chose mode to collect embeddings",
    )
    parser.add_argument(
        "--input-path",
        default="datasets/mutants_min:13.71_hev:15.82.json",
        help="Path to input JSON file",
    )

    args = parser.parse_args()
    print(f"Starting in mode {args.mode}")

    classificator = Classificator()
    collect_embeddings(classificator, args.mode, args.input_path)
