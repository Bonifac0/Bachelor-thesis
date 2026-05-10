import re
import matplotlib.pyplot as plt
import numpy as np

"""
python -m src.visualization.training_log_viz
"""

METRICS = {"acc": "Accuracy", "f1": "F1"}


def parse_training_log(file_name):
    """Parses the entire log into a structured dictionary."""
    epoch_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/\d+ \| Train Loss: (?P<train_loss>[\d.]+) \| Val Loss: (?P<val_loss>[\d.]+) \| Val Precision: (?P<precision>[\d.]+) \| Recall: (?P<recall>[\d.]+) \| F1: (?P<f1>[\d.]+) \| Accuracy: (?P<acc>[\d.]+)"
    )
    group_pattern = re.compile(
        r"Group\s+(?P<range>[\d\s-]+) \| Samples:\s+(?P<samples>\d+) \| Acc: (?P<acc>[\d.]+) \| F1: (?P<f1>[\d.]+)"
    )
    aa_pattern = re.compile(
        r"AA\s+(?P<acid>\w) \| Samples:\s+(?P<samples>\d+) \| Acc: (?P<acc>[\d.]+) \| F1: (?P<f1>[\d.]+)"
    )

    results = {"epochs": {}, "protein_groups": {}, "amino_acids": {}}

    # Assuming files are in training_logs/ directory
    try:
        with open(f"training_logs/{file_name}.log", "r") as f:
            for line in f:
                # Epochs
                if m := epoch_pattern.search(line):
                    d = m.groupdict()
                    results["epochs"][int(d.pop("epoch"))] = {
                        k: float(v) for k, v in d.items()
                    }
                    continue
                # Groups
                if m := group_pattern.search(line):
                    d = m.groupdict()
                    clean_range = "".join(d["range"].split())
                    results["protein_groups"][clean_range] = {
                        k: (int(v) if k == "samples" else float(v))
                        for k, v in d.items()
                        if k != "range"
                    }
                    continue
                # AA
                if m := aa_pattern.search(line):
                    d = m.groupdict()
                    results["amino_acids"][d["acid"]] = {
                        k: (int(v) if k == "samples" else float(v))
                        for k, v in d.items()
                        if k != "acid"
                    }
    except FileNotFoundError:
        print(f"Error: training_logs/{file_name}.log not found.")

    return results


def plot_group_comparison(runs_data, output_pdf):
    """
    Plots sequence length histogram for any number of runs.
    :param runs_data: List of tuples [(data_dict, "Label Name"), ...]
    """
    if not runs_data:
        return

    # Extract all unique group keys from the first run and sort them
    first_dict = runs_data[0][0]
    sorted_keys = sorted(
        first_dict["protein_groups"].keys(), key=lambda x: int(x.split("-")[0])
    )

    x = np.arange(len(sorted_keys))
    num_runs = len(runs_data)

    # Calculate bar width based on number of runs (total available width per group is approx 0.8)
    total_width = 0.8
    width = total_width / num_runs

    fig, ax = plt.subplots(figsize=(max(12, num_runs * 3), 7))

    # Plot each run
    for i, (data_dict, label) in enumerate(runs_data):
        values = [1.0 - data_dict["protein_groups"][k]["acc"] for k in sorted_keys]

        # Calculate offset so bars are centered around the tick
        offset = (i - (num_runs - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=label, edgecolor="black")

        # Value labels (optional: only show if fewer than 4 runs to avoid clutter)
        if num_runs <= 4:
            for rect in rects:
                ax.annotate(
                    f"{rect.get_height():.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )

    # Formatting
    ax.set_ylabel("Error Rate", fontsize=18)
    # ax.set_ylim(0, 1.1)
    ax.set_title(
        "Performance Comparison by Protein Length Group",
        fontsize=24,
    )
    ax.set_xlabel("Length Group", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_keys, rotation=45, fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()
    print(f"Plot successfully saved to {output_pdf}")


def plot_aa_comparison(runs_data, output_pdf):
    """
    Plots amino acid performance comparison for any number of runs.
    :param runs_data: List of tuples [(data_dict, "Label Name"), ...]
    """
    if not runs_data:
        return

    # Extract all unique AA keys from the first run and sort alphabetically
    first_dict = runs_data[0][0]
    sorted_aa = sorted(first_dict["amino_acids"].keys())

    x = np.arange(len(sorted_aa))
    num_runs = len(runs_data)

    # Calculate bar width based on number of runs
    total_width = 0.8
    width = total_width / num_runs

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each run
    for i, (data_dict, label) in enumerate(runs_data):
        # We use .get(aa, 0) in case one log is missing an AA that another has
        values = [
            1.0 - data_dict["amino_acids"].get(aa, {}).get("acc", 0) for aa in sorted_aa
        ]

        offset = (i - (num_runs - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=label, edgecolor="black")

        # Add value labels for readability (only if 3 or fewer runs)
        if num_runs <= 3:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )

    # Formatting
    ax.set_ylabel("Error Rate", fontsize=18)
    ax.set_title("Performance Comparison by Amino Acid", fontsize=24)
    ax.set_xlabel("Amino Acid", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_aa, fontsize=18)
    # ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()
    print(f"AA comparison plot saved to {output_pdf}")


if __name__ == "__main__":
    runs = [  # acc_group_bacic_vs_len
        (parse_training_log("basic"), "Basic"),
        (parse_training_log("length"), "With Length Feature"),
    ]
    plot_group_comparison(runs, "graphs/acc_group_bacic_vs_len.pdf")

    runs = [  # acc_len_dif_layers
        (parse_training_log("basic"), "Basic"),
        (parse_training_log("HL_16"), "One Hidden Layer"),
        (parse_training_log("2HL_64_16"), "Two Hidden Layers"),
        (parse_training_log("3HL_64_32_16"), "Three Hidden Layers"),
    ]
    plot_aa_comparison(runs, "graphs/acc_AA_dif_layers.pdf")
