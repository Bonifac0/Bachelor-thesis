from src.predictor import Classificator
from src.helpers.levenshtein import levenshtein
from src.helpers.print_eta import ETA
import numpy as np
import random
import json
import time


"""
to run:
python -m src.training.reverse_mutation_generator
"""

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_single_mutants(sequence: str, n: int) -> list[str]:
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


def reverse_bulk(
    mdl: Classificator,
    baseline: str,
    bulk_threshold: float,
    batchsize: int = 20,
    max_cycles: int = 30,
) -> tuple[str, int] | None:
    """
    Adding mutation to move away from thermoclass.
    Return mutant and number of mutation.
    """
    best_mutant = baseline
    best_score: float = mdl.classify([("", baseline)])[0][3]
    print(f"baseline: {best_score}")

    for cycle in range(min(max_cycles, len(baseline))):  # max cycles
        better_mutants = random_single_mutants(best_mutant, batchsize)
        props = mdl.classify([("", i) for i in better_mutants])
        best_index = np.argmin([i[3] for i in props])  # least hyperthermophilic
        print(props[best_index][3])

        if props[best_index][3] <= bulk_threshold:
            num_mutation = levenshtein(best_mutant, baseline)

            return (best_mutant, num_mutation)
        else:
            if props[best_index][3] < best_score:
                best_mutant = better_mutants[best_index]
                best_score = props[best_index][3]

    return None  # isnt possible to mutate enough


def reverse_cut(
    mdl: Classificator,
    heavy_mutant: str,
    baseline: str,
    cut_threshold: float,
) -> tuple[str, int]:
    """
    Cutting mutation while keeping low thermoclass
    """
    minimal_mutant = heavy_mutant
    for cycle in range(levenshtein(heavy_mutant, baseline)):
        lighter_mutants = single_revert(baseline, minimal_mutant)
        props = mdl.classify([("", i) for i in lighter_mutants])
        best_index = np.argmin([i[3] for i in props])  # least hyperthermophilic

        if props[best_index][3] <= cut_threshold:
            minimal_mutant = lighter_mutants[best_index]

        else:
            break
    num_mutation = levenshtein(minimal_mutant, baseline)
    return (minimal_mutant, num_mutation)


def reverse_mutate(
    mdl: Classificator,
    baseline: str,
    cut_threshold: float = 0.05,
    bulk_threshold: float = 0.02,
) -> tuple[str, int, int] | None:
    """
    Create less hyperthermophilic mutant of baseline (hyperthermo) protein,
    based on given model.

    Add mutation until bulk_threshold is reach and than
    least important changes are removed until the cut_threshold is reached.
    """

    heavy_mutant = reverse_bulk(mdl, baseline, bulk_threshold)
    if heavy_mutant is None:
        return None
    minimal_mutant = reverse_cut(mdl, heavy_mutant[0], baseline, cut_threshold)

    return minimal_mutant[0], heavy_mutant[1], minimal_mutant[1]


def main(mdl: Classificator):
    protein_list: list[dict] = [
        {
            "prot_id": "A0A1M6DL67",
            "original_hyper": "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
        },
        {
            "prot_id": "A0A1X6YLE1",
            "original_hyper": "AYFLRAETGAATPNKWPWGDVAIIADVRMEDDVIKKFRA",
        },
    ]

    protein_count = len(protein_list)
    eta = ETA(protein_count)
    not_mutated = 0
    sum_min = 0
    sum_hev = 0
    output = []

    print()
    # print(
    #     f"Mutated protein {0}/{protein_count} | Not mutated 0 | ETA: ?",
    #     end="\r",
    # )
    try:
        for idx, entry in enumerate(protein_list):
            mut_pack = reverse_mutate(mdl, entry["original_hyper"])
            if mut_pack is not None:
                mutant, st_hev, st_min = mut_pack
                entry["mutant_mezo"] = mutant
                output.append(entry)
                sum_hev += st_hev
                sum_min += st_min
            else:
                not_mutated += 1

            # print(
            #     f"Mutated protein {idx + 1}/{protein_count} | Not mutated {not_mutated} | min:{sum_min / (idx + 1 - not_mutated):.2f} hev:{sum_hev / (idx + 1 - not_mutated):.2f} {eta.print_eta(idx + 1)}",
            #     end="\r",
            # )
        idx += 1
    finally:
        print()
        print(f"Mutated protein {idx}/{protein_count}")

        print(f"Not mutated: {not_mutated}")

        t = time.strftime("%T")
        with open(
            f"reverse_mutants{t}_min:{sum_min / (idx - not_mutated):.2f}_hev:{sum_hev / (idx - not_mutated):.2f}.json",
            "w",
        ) as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    classificator = Classificator()
    main(classificator)
