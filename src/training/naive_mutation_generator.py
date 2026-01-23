from src.predictor import Classificator
from src.helpers.levenshtein import levenshtein
from src.helpers.print_eta import ETA
import numpy as np
import random
import json
import time


"""
to run:
python -m src.training.naive_mutation_generator

because pyhon need to load packages
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


def bulk(  # TODO OPTIONAL add option to overcome local minimum
    mdl: Classificator,
    baseline: str,
    top_threshold: float = 0.9,
    batchsize: int = 20,
    max_cycles: int = 30,
) -> tuple[str, int] | None:
    """
    Adding mutation to move to thermoclass.
    Return mutant and number of mutation.
    """
    best_mutant = baseline
    best_score: float = mdl.classify([("", baseline)])[0][3]
    # print(f"baseline: {best_score}")

    for cycle in range(min(max_cycles, len(baseline))):  # max cycles
        better_mutants = random_single_mutants(best_mutant, batchsize)
        props = mdl.classify([("", i) for i in better_mutants])
        best_index = np.argmax([i[3] for i in props])  # most hyperthermophilic

        if props[best_index][3] >= top_threshold:
            num_mutation = levenshtein(best_mutant, baseline)
            # print(f"number of mutations: {num_mutation}")
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
    # print(f"final number of mutations: {num_mutation}")
    return (minimal_mutant, num_mutation)


def mutate(
    mdl: Classificator,
    baseline: str,
    bottom_threshold: float = 0.75,
    top_threshold: float = 0.85,
) -> tuple[str, int, int] | None:
    """
    Create more hyperthermophilic mutant of baseline protein,
    based on given model.

    Add mutation until top_threshold is reach and than
    least important changes are removed until the bottom_threshold is reached.
    """

    heavy_mutant = bulk(mdl, baseline, top_threshold)
    if heavy_mutant is None:
        return None
    minimal_mutant = cut(mdl, heavy_mutant[0], baseline, bottom_threshold)
    # print(baseline)
    # print(minimal_mutant[0])
    # print(heavy_mutant[1], minimal_mutant[1])

    return minimal_mutant[0], heavy_mutant[1], minimal_mutant[1]


def collect_proteins(data: dict, offset: str = "") -> list:
    """
    Return list of proteins that fit conditions.
    Will start collecting after offset (famID_protID),
    if empty, start from beginning
    """
    dropped = 0
    protein_list = []
    collect: bool = offset == ""
    if collect:
        print("Collecting from beginning")
    else:
        offset_fam, offset_prot = offset.split("_")
        print(f"Collecting after {offset}")

    for fam, entries in data.items():
        for prot_id, entry in entries.items():
            if collect:
                if entry["temp"] <= 35:  # only psy and mezo
                    if len(entry["domain"]) > 500:
                        dropped += 1
                        continue
                    prot: dict = {"prot_id": prot_id}
                    prot["domain"] = entry["domain"]
                    protein_list.append(prot)
            elif fam == offset_fam and prot_id == offset_prot:
                collect = True

    if not collect:
        print("Offset protein not found")
        exit()
    print(f"Dropped becaouse domain len: {dropped}")
    print(f"Collected {len(protein_list)} proteins")
    return protein_list


def main(mdl: Classificator):
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    protein_list = collect_proteins(data, START_AFTER_PROT)

    protein_count = len(protein_list)
    eta = ETA(protein_count)
    not_mutated = 0
    sum_min = 0
    sum_hev = 0
    output = []

    print()
    print(
        f"Mutated protein {0}/{protein_count} | Not mutated 0 | ETA: ?",
        end="\r",
    )
    try:
        for idx, entry in enumerate(protein_list):
            mut_pack = mutate(mdl, entry["domain"])
            if mut_pack is not None:
                mutant, st_hev, st_min = mut_pack
                entry["mutant"] = mutant
                output.append(entry)
                sum_hev += st_hev
                sum_min += st_min
            else:
                not_mutated += 1

            print(
                f"Mutated protein {idx + 1}/{protein_count} | Not mutated {not_mutated} | min:{sum_min / (idx + 1 - not_mutated):.2f} hev:{sum_hev / (idx + 1 - not_mutated):.2f} {eta.print_eta(idx + 1)}",
                end="\r",
            )
    finally:
        print()
        print(f"Mutated protein {idx}/{protein_count}")

        print(f"Not mutated: {not_mutated}")

        t = time.strftime("%T")
        with open(
            f"datasets/mutants{t}_min:{sum_min / (idx + 1 - not_mutated):.2f}_hev:{sum_hev / (idx + 1 - not_mutated):.2f}.json",
            "w",
        ) as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    classificator = Classificator()

    INPUT_PATH = "datasets/processed_dataset.json"
    # INPUT_PATH = "test_mut_inp.json"
    OUTPUT_PATH = "datasets/mutants"  # the stats will be appendet after this

    # everithing before this prot (included) will be scipped
    # famID_protID
    # if empty, start from beginning
    START_AFTER_PROT = "PF00791_A9GXW8"  # for subsequent runs of this script
    # START_AFTER_PROT = ""

    main(classificator)
