import random
from src.predictor import Classificator
from src.helpers.importance_vis import make_importance_general
from src.training.run_model import ModelRunner
from src.training.reverse_mutation_generator import reverse_mutate
from src.helpers.captum_embedding import get_captum_embedding
from src.helpers.print_eta import ETA

# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import json


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


def use_chaotic_mutations(
    classificator, baseline, baseline_score, iterations=50
) -> np.ndarray:
    percetage = 0.1
    mutations = generate_random_mutations(baseline, percetage, iterations)
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
        # print(indices)
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
    # print()
    # print(real_score)
    return real_score


def aggregate_embedding(ig_embedding: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L1 aggregation for an Integrated Gradients embedding attribution.
    """
    # TODO read https://arxiv.org/html/2507.18043v1?utm_source=chatgpt.com
    # source acording to chatbot

    l1 = np.abs(ig_embedding).sum(axis=-1)
    return l1 / (l1.max() + eps)


if __name__ == "__main__":
    classificator = Classificator()
    runner = ModelRunner(classificator)

    proteins = [
        # {
        #     "prot_id": "alice",
        #     "domain": "DRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAG",
        #     "mutant": "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
        # },
        # {
        #     "prot_id": "bob",
        #     "domain": "ALQLRAETGAATPADWHWGDVAIIADNRTEADVIRQFRA",
        #     "mutant": "AYFLRAETGAATPNKWPWGDVAIIADVRMEDDVIKKFRA",
        # },
        {
            "prot_id": "A0A4R3J6H1",
            "domain": "REIRKTFQVALENHKSGVPLTWRDKKTGAATTVTPVLTYKAASGAFCRTYRQSITLNGKTHLYPGVACRESRLKWVIPRLAQLVGNTSRFTVINHVKLKGAKKDTKYVQRWQCAVDGTERVRVLAGTFDTYKVECKRFSPTFRFYQKRTWYYAPEIGQYVRREDYYKYPGKTY",
            "mutant": "REIRKTFQVALENHKSGVPLTWRDKKTGAYTTVTFVLTYKARSGAFCRTYRYSITLNGKTYLYRGVACRESRLKWVIPRLAQLVGNTSRFTVINVVKLKGAKKDTKYVQRWQCAVDGTERVRVLAGTFDTYKVECKRFSPTFRFYQKRTWYYACEIGQYVRREDYYKYPGKTY",
        },
    ]
    # with open("selected_dom_mut_pair.json", "r") as f:
    #     proteins = json.load(f)

    protein_count = len(proteins)
    eta = ETA(protein_count)
    print()
    print(
        f"Tested protein {0}/{protein_count} | ETA: ?",
        end="\r",
    )

    counter_all = 0
    counter_correct = 0

    for idx, protein in enumerate(proteins):
        probability = (
            classificator.classify([("", protein["domain"])])[0][3],
            classificator.classify([("", protein["mutant"])])[0][3],
        )

        pred_mut: np.ndarray = runner.predict_importance(protein["mutant"])
        pred_dom: np.ndarray = runner.predict_importance(protein["domain"])

        difference = [
            1 if d != m else 0 for d, m in zip(protein["domain"], protein["mutant"])
        ]

        for j, (d, m) in enumerate(zip(protein["domain"], protein["mutant"])):
            if d != m:
                counter_all += 1
                if pred_dom[j] < 0.5 and pred_mut[j] > 0.5:
                    counter_correct += 1

        # real_decrease: np.ndarray = use_chaotic_mutations(
        #     classificator, protein["mutant"], probability[1]
        # )

        # mut_embedding = get_captum_embedding(classificator, protein["mutant"])
        # dom_embedding = get_captum_embedding(classificator, protein["domain"])

        # mut_cap_importance = aggregate_embedding(mut_embedding)
        # dom_cap_importance = aggregate_embedding(dom_embedding)

        data = np.row_stack(
            [
                pred_mut,
                pred_dom,
                # mut_cap_importance,
                # dom_cap_importance,
                # real_decrease,
            ]
        )

        labels = [
            "Predictor mutant",
            "Predictor domain",
            # "Captum relative mutant",
            # "Captum relative domain",
            # "Real decrease",
        ]

        make_importance_general(
            protein, data, probability, labels, outdir="test_importance/full"
        )

        # data_only_mut = np.row_stack(
        #     [
        #         pred_mut,
        #         mut_cap_importance,
        #         # real_decrease,
        #     ]
        # )

        # labels_only_mut = [
        #     "Predictor mutant",
        #     "Captum relative mutant",
        #     # "Real decrease",
        # ]
        # make_importance_general(
        #     protein,
        #     data_only_mut,
        #     probability,
        #     labels_only_mut,
        #     outdir="test_importance/only_mut",
        # )

        print(
            f"Tested protein {idx + 1}/{protein_count} {eta.print_eta(idx + 1)}",
            end="\r",
        )
    print()
    eta.print_elapsed()
    print(counter_correct / counter_all)
