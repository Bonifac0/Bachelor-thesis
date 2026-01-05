from src.predictor import Classificator
from src.heplers.levenshtein import levenshtein
import numpy as np


"""
to run:
python -m src.training.naive_mutation_generator

because pyhon need to load packages
"""


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


def bulk(
    mdl: Classificator, baseline: str, top_threshold: float = 0.9, batchsize: int = 10
) -> str:
    """
    Adding mutation to move to thermoclass
    """
    for cycle in range(20):  # max cycles
        pass
    return "MGRGKSKWFNNRKGYG"


def cut(  # TODO check if work corectlly
    mdl: Classificator,
    heavy_mutant: str,
    baseline: str,
    bottom_threshold: float = 0.8,
) -> str:
    """
    Cutting mutation while keaping thermoclass
    """
    minimal_mutant = heavy_mutant
    for cycle in range(levenshtein(heavy_mutant, baseline)):
        lighter_mutants = single_revert(baseline, minimal_mutant)
        props = mdl.classify([("", i) for i in lighter_mutants])
        best_index = np.argmax([i[3] for i in props])  # most hyperthermophilic

        if props[best_index][3] >= bottom_threshold:
            minimal_mutant = lighter_mutants[best_index]
            print([props[i][3] for i in range(len(props))], best_index)
        else:
            break
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

    for baseline in example_inp:
        print(f"baseline: {mdl.classify([baseline])[0][3]}")
        heavy_mutant = bulk(mdl, baseline[1])
        minimal_mutant = cut(mdl, heavy_mutant, baseline[1])
        print(minimal_mutant)
    # outputs = mdl.classify(example_inp)


if __name__ == "__main__":
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    main(classificator)
