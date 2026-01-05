from src.predictor import Classificator
from src.heplers.levenshtein import levenshtein
import numpy as np
import random


"""
to run:
python -m src.training.naive_mutation_generator

because pyhon need to load packages
"""

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_single_mutants(sequence: str, n: int = 10) -> list[str]:
    seen = set()
    mutants = []

    while len(mutants) < n:
        pos = random.randrange(len(sequence))
        new_aa = random.choice([aa for aa in AMINO_ACIDS if aa != sequence[pos]])

        mutant = sequence[:pos] + new_aa + sequence[pos + 1 :]
        if mutant not in seen:
            seen.add(mutant)
            mutants.append(mutant)

    return mutants


def single_revert(baseline: str, modified: str) -> list[str]:
    if len(baseline) != len(modified):
        raise ValueError("Sequences must be the same length")

    diffs = [i for i in range(len(baseline)) if baseline[i] != modified[i]]
    variants = []

    for i in diffs:
        seq = list(modified)
        seq[i] = baseline[i]
        variants.append("".join(seq))

    return variants


def bulk(  # TODO OPTIONAL add option to overcome local minimum
    mdl: Classificator, baseline: str, top_threshold: float = 0.9, batchsize: int = 10
) -> str | None:
    """
    Adding mutation to move to thermoclass
    """
    best_mutant = baseline
    best_score: float = mdl.classify([("", baseline)])[0][3]
    print(f"baseline: {best_score}")

    for cycle in range(min(20, len(baseline))):  # max cycles
        better_mutants = random_single_mutants(best_mutant, batchsize)
        props = mdl.classify([("", i) for i in better_mutants])
        best_index = np.argmax([i[3] for i in props])  # most hyperthermophilic

        if props[best_index][3] >= top_threshold:
            print(f"number of cycles: {cycle}")
            print(f"number of mutations: {levenshtein(best_mutant, baseline)}")
            return best_mutant
        else:
            if props[best_index][3] > best_score:
                best_mutant = better_mutants[best_index]
                best_score = props[best_index][3]
                print(props[best_index][3])
            else:
                print("better not found in this iteration")
            # print([props[i][3] for i in range(len(props))], best_index)

        # print(f"best score: {best_score}")
    return None  # isnt possible to mutate enough


def cut(  # TODO check if work corectlly
    mdl: Classificator,
    heavy_mutant: str,
    baseline: str,
    bottom_threshold: float = 0.8,
) -> str:
    """
    Cutting mutation while keeping thermoclass
    """
    minimal_mutant = heavy_mutant
    for cycle in range(levenshtein(heavy_mutant, baseline)):
        lighter_mutants = single_revert(baseline, minimal_mutant)
        props = mdl.classify([("", i) for i in lighter_mutants])
        best_index = np.argmax([i[3] for i in props])  # most hyperthermophilic

        if props[best_index][3] >= bottom_threshold:
            minimal_mutant = lighter_mutants[best_index]
            print(props[best_index][3])
            # print([props[i][3] for i in range(len(props))], best_index)
        else:
            break
    print(f"final number of mutations: {levenshtein(minimal_mutant, baseline)}")
    return minimal_mutant


def main(mdl: Classificator):
    example_inp = [
        (
            "term",
            "MQRGKVKWFNNEKGYG",
        ),
        # (
        #     "mezo",
        #     "MLEGKVKWFNSEKGFGFIEVEG",
        # ),
    ]
    # MRRGKVVWWNIEKGYG

    for baseline in example_inp:
        heavy_mutant = bulk(mdl, baseline[1], 0.8)
        if heavy_mutant is None:
            print(f"protein '{baseline[0]}' cannot be mutate enough")
            continue
        minimal_mutant = cut(mdl, heavy_mutant, baseline[1], 0.65)
        print(baseline[1])
        print(minimal_mutant)
    # outputs = mdl.classify(example_inp)


if __name__ == "__main__":
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    main(classificator)
