from src.predictor import Classificator
from src.heplers.levenshtein import levenshtein
import numpy as np
import random
import json


"""
to run:
python -m src.training.naive_mutation_generator

because pyhon need to load packages
"""

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_single_mutants(sequence: str, n: int = 50) -> list[str]:
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
    mdl: Classificator,
    baseline: str,
    top_threshold: float = 0.9,
    batchsize: int = 10,
    max_cycles: int = 20,
) -> tuple[str, int] | None:
    """
    Adding mutation to move to thermoclass.
    Return mutant and number of mutation.
    """
    best_mutant = baseline
    best_score: float = mdl.classify([("", baseline)])[0][3]
    print(f"baseline: {best_score}")

    for cycle in range(min(max_cycles, len(baseline))):  # max cycles
        better_mutants = random_single_mutants(best_mutant, batchsize)
        props = mdl.classify([("", i) for i in better_mutants])
        best_index = np.argmax([i[3] for i in props])  # most hyperthermophilic

        if props[best_index][3] >= top_threshold:
            num_mutation = levenshtein(best_mutant, baseline)
            print(f"number of mutations: {num_mutation}")
            return (best_mutant, num_mutation)
        else:
            if props[best_index][3] > best_score:
                best_mutant = better_mutants[best_index]
                best_score = props[best_index][3]
                # print(props[best_index][3])
            # else:
            # print("better not found in this iteration")
            # print([props[i][3] for i in range(len(props))], best_index)

        # print(f"best score: {best_score}")
    return None  # isnt possible to mutate enough


def cut(  # TODO check if work corectlly
    mdl: Classificator,
    heavy_mutant: str,
    baseline: str,
    bottom_threshold: float = 0.8,
) -> tuple[str, int]:
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
            # print(props[best_index][3])
            # print([props[i][3] for i in range(len(props))], best_index)
        else:
            break
    num_mutation = levenshtein(minimal_mutant, baseline)
    print(f"final number of mutations: {num_mutation}")
    return (minimal_mutant, num_mutation)


def mutate(
    mdl: Classificator,
    baseline: str,
    bottom_threshold: float = 0.8,
    top_threshold: float = 0.9,
) -> dict | None:
    """
    Create more hyperthermophilic mutant of baseline protein,
    based on given model.

    Add mutation until top_threshold is reach and than
    least important changes are removed until the bottom_threshold is reached.
    """
    output: dict = {}
    output["sequence"] = baseline

    heavy_mutant = bulk(mdl, baseline, top_threshold)
    if heavy_mutant is None:
        return None
    minimal_mutant = cut(mdl, heavy_mutant[0], baseline, bottom_threshold)
    print(baseline)
    print(minimal_mutant[0])
    output["mutant"] = minimal_mutant[0]
    output["mut_stat"] = (heavy_mutant[1], minimal_mutant[1])

    return output


def main(mdl: Classificator):
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    output = []
    for _ in range(15):  # repeat to get more mutants
        for fam_id, fam in data.items():
            for prot_id, prot in fam.items():
                if True:  # TODO condition of protein qualities
                    protein = mutate(mdl, prot["domain"])
                    if protein is None:
                        print(f"protein '{prot_id}' cannot be mutate enough")
                    else:
                        protein["prot_id"] = prot_id
                        output.append(protein)

    with open(OUTPUT_PATH, "a") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    classificator = Classificator()

    # INPUT_PATH = "datasets/processed_dataset.json"
    INPUT_PATH = "test_mut_inp.json"
    OUTPUT_PATH = "test_mutation.json"

    main(classificator)
    # usefull for diff indices
    # [i for i in range(len(baseline)) if baseline[i] != modified[i]]
