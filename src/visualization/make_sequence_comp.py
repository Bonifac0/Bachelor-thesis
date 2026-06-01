from src.helpers.importance_vis import make_importance_diff
from src.training.run_model import ModelRunner
from src.helpers.captum_attribution import get_captum_attribution
from src.helpers.print_eta import ETA
from src.evaluations.importance_tester import aggregate_log_sigmoid
import numpy as np
import json


"""
python -m src.visualization.make_sequence_comp
"""


def main():
    runner = ModelRunner("2HL_64_16")

    # grep -C 5 "A0A290QAY7" datasets/processed_dataset.json
    with open("output.json", "r") as f:
        proteins = json.load(f)

    protein_count = len(proteins)
    eta = ETA(protein_count)
    print()
    print(
        f"Tested protein {0}/{protein_count} | ETA: ?",
        end="\r",
    )

    for idx, protein in enumerate(proteins):
        probability = (
            runner.classificator.classify([("", protein["domain"])])[0][3],
            runner.classificator.classify([("", protein["mutant"])])[0][3],
        )

        mut_attribution = get_captum_attribution(
            runner.classificator, protein["mutant"]
        )
        dom_attribution = get_captum_attribution(
            runner.classificator, protein["domain"]
        )

        data = np.row_stack(
            [
                runner.predictor_inference(mut_attribution),
                aggregate_log_sigmoid(mut_attribution),
                runner.predictor_inference(dom_attribution),
                aggregate_log_sigmoid(dom_attribution),
            ]
        )

        labels = [
            "Predictor mutant",
            "Abs sum mutant",
            "Predictor original",
            "Abs sum original",
        ]

        make_importance_diff(protein, data, probability, labels)

        print(
            f"Tested protein {idx + 1}/{protein_count} {eta.print_eta(idx + 1)}",
            end="\r",
        )
    print()
    eta.print_elapsed()


if __name__ == "__main__":
    main()
