import random
from src.predictor import Classificator
from src.helpers.importance_vis import make_importance_hyperthermo_compare
from src.training.run_model import ModelRunner


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

    # domain = ("DOM", "DRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAG")
    # mutant = ("MUT", "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG")
    domain = (
        "DOM_LONG",
        "ALRTHTEKDCYTPGGWRPGDEVLVPDEIADDDVETRYGESIGDQFPHFIGSSTRGTIDFRRWAEGSWVVFFSQPGTFSSVCTTEMGMFVRDQELFEERGIKLIALAKEDLGEQQSWLREIERLYGARVDFPSVEDPTGMLSTAFGMTTDREIAPEPARITFIMDPHRVIRMI",
    )
    mutant = (
        "MUT_LONG",
        "ALRTHTEKCCYTPGGWRCGDWVLVPDEIADDDVETRYGESIGDTFPHFIGVSTRGTIWFRRWAEGSWVVFFSQPGTFSSVCTFEMMMFVRDVELFEERGIKLIALAKEDLGEQQSWLREIERLYGARVDFPSVEDYTGMLSTAFGMTTYRVIYPEPARITFIMDPHRVIRMI",
    )
    assert len(domain[1]) == len(mutant[1])
    domain_score = classificator.classify([domain])[0][3]
    mutant_score = classificator.classify([mutant])[0][3]
    print(f"Baseline: {domain_score}")
    print(f"Mutant: {mutant_score}")

    variants, indices = single_revert(domain[1], mutant[1])

    mutations = [("", mut) for mut in variants]
    probability = [i[3] for i in classificator.classify(mutations)]

    counter = 0
    indices.append(-1)

    data = []
    print(indices)
    for idx, mut in enumerate(mutant[1]):
        if idx == indices[counter]:
            score = mutant_score - probability[counter]
            counter += 1
        else:
            score = 0
        data.append(score)

        # print(f"{mut}: {score}")

    predicted = runner.predict_importance(mutant[1])

    make_importance_hyperthermo_compare(mutant, predicted, data, mutant_score)
