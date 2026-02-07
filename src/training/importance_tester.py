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


if __name__ == "__main__":
    classificator = Classificator()
    runner = ModelRunner(classificator)

    hypertermo = (
        "MUT",
        "ETLLIENFKSIRRLELKLRPGVNLLVGPNASGKTNILEAIYSFHKALFPPGVAQNSILATLAYVLEG",
    )

    hyperthermo_score = classificator.classify([hypertermo])[0][3]
    print(f"Hyperthermo: {hyperthermo_score}")
    predicted = runner.predict_importance(hypertermo[1])

    data = []

    for _ in range(5):
        sinthetic_mezo, _, _ = reverse_mutate(classificator, hypertermo[1], 0.1, 0.1)
        if single_revert is None:
            print("didnt make it")
            continue
        variants, indices = single_revert(sinthetic_mezo, hypertermo[1])

        mutations = [("", mut) for mut in variants]
        probability = [i[3] for i in classificator.classify(mutations)]

        counter = 0
        indices.append(-1)

        single_data = []
        print(indices)
        for idx, mut in enumerate(hypertermo[1]):
            if idx == indices[counter]:
                score = hyperthermo_score - probability[counter]
                counter += 1
            else:
                score = 0
            single_data.append(score)
        # print(single_data)
        data.append(single_data)

    real_score = np.mean(data, axis=0)
    print()
    print(real_score)

    make_importance_hyperthermo_compare(
        hypertermo, predicted, real_score, hyperthermo_score
    )
