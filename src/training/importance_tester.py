import random
from src.predictor import Classificator
import numpy as np


"""
to run:
python -m src.training.importance_tester
"""


def random_single_mutations(protein):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mutated_sequences = []

    for i, aa in enumerate(protein):
        choices = [x for x in amino_acids if x != aa]
        new_aa = random.choice(choices)
        mutated_sequences.append(protein[:i] + new_aa + protein[i + 1 :])

    return mutated_sequences


if __name__ == "__main__":
    classificator = Classificator()

    protein = "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG"
    baseline = classificator.classify([("", protein)])[0][3]
    print(f"Baseline: {baseline}")

    data = [[] for _ in range(len(protein))]
    for i in range(30):
        mutations = [("", mut) for mut in random_single_mutations(protein)]
        # print(mutations)
        probability = [i[3] for i in classificator.classify(mutations)]

        for idx, p in enumerate(probability):
            # print(f"{protein[idx]}: {baseline - p}")
            data[idx].append(baseline - p)
        print(f"Done {i}")
    for idx, i in enumerate(data):
        print(f"{protein[idx]}: {np.mean(i)}")
