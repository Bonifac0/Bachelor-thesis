import json
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("agg")

from src.predictor import Classificator
from src.training.run_model import ModelRunner
from src.helpers.print_eta import ETA

"""
python -m src.evaluations.make_acc_len_histogram
"""


def calculate_len_stats(output_json="test_importance/len_stats.json"):
    classificator = Classificator()
    runner = ModelRunner(classificator)
    with open("selected_dom_mut_pair.json", "r") as f:
        proteins = json.load(f)

    # Specify groups range manually: [(low, high), ...]
    # ranges = [(0, 40), (41, 80), (81, 120), (121, 160), (161, 250)]
    ranges = [(0, 25), (26, 28), (31, 33)]

    groups = [[] for _ in ranges]
    bin_labels = [f"{low}-{high}" for low, high in ranges]
    counts = []
    bin_accuracies = []
    num_of_mutations = []

    for p in proteins:
        p_len = len(p["domain"])
        for idx, (low, high) in enumerate(ranges):
            if low <= p_len <= high:
                groups[idx].append(p)
                break

    for gr in groups:
        print(f"Group count {len(gr)}")

    total_proteins = sum(len(g) for g in groups)
    processed_count = 0
    eta = ETA(total_proteins)

    print(f"Starting length stats calculation for {total_proteins} proteins...")

    for i, group in enumerate(groups):
        counts.append(len(group))

        if not group:
            bin_accuracies.append(0)
            num_of_mutations.append(0)
            continue

        counter_all = 0
        counter_correct = 0

        for protein in group:
            pred_mut = runner.predict_importance(protein["mutant"])
            pred_dom = runner.predict_importance(protein["domain"])

            for j, (d, m) in enumerate(zip(protein["domain"], protein["mutant"])):
                if d != m:
                    counter_all += 1
                    if pred_dom[j] < 0.5 and pred_mut[j] > 0.5:
                        counter_correct += 1

            processed_count += 1
            print(
                f"Processed {processed_count}/{total_proteins} {eta.print_eta(processed_count)}",
                end="\r",
            )

        accuracy = counter_correct / counter_all if counter_all > 0 else 0
        bin_accuracies.append(accuracy)
        num_of_mutations.append(counter_all)

    results = {
        "bin_labels": bin_labels,
        "bin_accuracies": bin_accuracies,
        "counts": counts,
        "num_of_mutations": num_of_mutations,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nStats saved to {output_json}")
    return results


def visualize_len_stats(input_json="test_importance/len_stats.json"):
    if not os.path.exists(input_json):
        print(f"Error: {input_json} not found. Run calculation first.")
        return

    with open(input_json, "r") as f:
        results = json.load(f)

    bin_labels = results["bin_labels"]
    bin_accuracies = results["bin_accuracies"]
    counts = results["counts"]
    num_of_mutations = results.get("num_of_mutations", [0] * len(bin_labels))

    print("\nSummary by length:")
    for label, acc, count in zip(bin_labels, bin_accuracies, counts):
        if count > 0:
            print(f"Bin {label}: Accuracy {acc:.4f}, Count {count}")

    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, bin_accuracies, color="skyblue")
    plt.xlabel("Protein Length Range")
    plt.ylabel("Accuracy")
    plt.title("Importance Prediction Accuracy by Protein Length")
    plt.ylim(0, 1.1)

    for i, acc in enumerate(bin_accuracies):
        if counts[i] > 0:
            plt.text(
                i,
                acc + 0.02,
                f"n={counts[i]}, acc={acc:.2f}, muts={num_of_mutations[i]}",
                ha="center",
                fontsize=6,
            )

    os.makedirs("test_importance", exist_ok=True)
    plt.savefig("test_importance/len_histogram.png")
    plt.close()
    print(f"Histogram saved to test_importance/len_histogram.png")


if __name__ == "__main__":
    # calculate_len_stats(classificator, runner)
    visualize_len_stats()
