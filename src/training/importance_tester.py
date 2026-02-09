import random
from src.predictor import Classificator
from src.helpers.importance_vis import make_importance_hyperthermo_compare
from src.training.run_model import ModelRunner
from src.training.reverse_mutation_generator import reverse_mutate
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


def single_revert(baseline: str, modified: str) -> tuple[list[str], list[int]]:
    """
    Returns list of mutants with one less mutated residum and indices of changes
    """
    if len(baseline) != len(modified):
        raise ValueError("Sequences must be the same length")

    diffs = [i for i in range(len(baseline)) if baseline[i] != modified[i]]
    variants = []

    for i in diffs:
        seq = list(modified)
        seq[i] = baseline[i]
        variants.append("".join(seq))

    return variants, diffs


def generate_random_mutations(dna: str, value: float, num_mutations: int = 10):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = len(dna)
    num_changes = max(1, int(length * value))

    mutations = []

    for _ in range(num_mutations):
        dna_list = list(dna)
        positions = random.sample(range(length), num_changes)

        for pos in positions:
            original = dna_list[pos]
            choices = [aa for aa in amino_acids if aa != original]
            dna_list[pos] = random.choice(choices)

        mutations.append("".join(dna_list))

    return mutations


def use_reverse_mutation(classificator, baseline, baseline_score):
    data = []
    for _ in range(5):
        sinthetic_mezo, _, _ = reverse_mutate(classificator, baseline, 0.1, 0.1)
        if single_revert is None:
            print("didnt make it")
            continue
        variants, indices = single_revert(sinthetic_mezo, baseline)

        mutations = [("", mut) for mut in variants]
        probability = [i[3] for i in classificator.classify(mutations)]

        counter = 0
        indices.append(-1)

        single_data = []
        print(indices)
        for idx, mut in enumerate(baseline):
            if idx == indices[counter]:
                score = baseline_score - probability[counter]
                counter += 1
            else:
                score = 0
            single_data.append(score)
        # print(single_data)
        data.append(single_data)

    real_score = np.mean(data, axis=0)
    print()
    print(real_score)
    return real_score


def use_chaotic_mutations(classificator, baseline, baseline_score):
    percetage = 0.1
    mutations = generate_random_mutations(baseline, percetage, 50)
    data = []
    for random_mutant in mutations:
        if single_revert is None:
            print("didnt make it")
            continue
        variants, indices = single_revert(random_mutant, baseline)

        mutations = [("", mut) for mut in variants]
        probability = [i[3] for i in classificator.classify(mutations)]

        counter = 0
        indices.append(-1)

        single_data = []
        print(indices)
        for idx, mut in enumerate(baseline):
            if idx == indices[counter]:
                score = (
                    baseline_score - probability[counter]
                ) / percetage  # normalization
                counter += 1
            else:
                score = 0
            single_data.append(score)
        # print(single_data)
        data.append(single_data)

    real_score = np.mean(data, axis=0)
    print()
    print(real_score)
    return real_score


if __name__ == "__main__":
    classificator = Classificator()
    runner = ModelRunner(classificator)

    hypertermo = (
        "MUT",
        "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
    )

    hyperthermo_score = classificator.classify([hypertermo])[0][3]
    print(f"Hyperthermo: {hyperthermo_score}")
    predicted = runner.predict_importance(hypertermo[1])

    # real_score = use_reverse_mutation(classificator, hypertermo[1], hyperthermo_score)
    real_score = use_chaotic_mutations(classificator, hypertermo[1], hyperthermo_score)

    make_importance_hyperthermo_compare(
        hypertermo, predicted, real_score, hyperthermo_score
    )
