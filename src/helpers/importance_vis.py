import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt

"""
python -m src.helpers.importance_vis
"""


def make_importance_all(
    protein: tuple[str, str],
    importance: list[list[float]],
    probability: list[float],
):
    labels = ["psychrophilic", "mesophilic", "thermophilic", "hyperthermophilic"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Importance for protein {protein[0]}", fontsize=16)
    axes = axes.flatten()

    for idx, imp in enumerate(importance):
        ax = axes[idx]
        x = range(len(protein[1]))

        ax.bar(x, imp)
        ax.set_xticks(x)
        ax.set_xticklabels(list(protein[1]))
        ax.set_xlabel("Amino Acid")
        ax.set_ylabel("Importance")
        ax.set_ylim([0, max(imp) * 1.2])
        ax.set_title(f"Importance for {labels[idx]} ({probability[idx]:.2f})")

        for i, val in enumerate(imp):
            ax.text(i, val + 0.01, f"{val:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"test_importance/{protein[0]}.png")


def make_importance_hyperthermo(
    protein: tuple[str, str],
    importance: list[float],
    probability: float,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Importance for protein {protein[0]}", fontsize=16)

    x = range(len(protein[1]))

    ax.bar(x, importance)
    ax.set_xticks(x)
    ax.set_xticklabels(list(protein[1]))
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Importance")
    ax.set_ylim((0, max(importance) * 1.2))
    ax.set_title(f"Importance for hyperthermophilic ({probability:.2f})")

    for i, val in enumerate(importance):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"test_importance/{protein[0]}_hyperthermo.png")


def make_importance_hyperthermo_compare(
    protein: tuple[str, str],
    importance_a: list[float],
    importance_b: list[float],
    probability: float,
    label_a: str = "Importance",
    label_b: str = "Real decrease",
):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    fig.suptitle(f"Importance comparison for protein {protein[0]}", fontsize=16)

    x = range(len(protein[1]))
    width = 0.4

    color_a = "tab:blue"
    color_b = "tab:orange"

    bars_a = ax1.bar(
        [i - width / 2 for i in x],
        importance_a,
        width=width,
        label=label_a,
        color=color_a,
    )

    bars_b = ax2.bar(
        [i + width / 2 for i in x],
        importance_b,
        width=width,
        label=label_b,
        color=color_b,
    )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(list(protein[1]))
    ax1.set_xlabel("Amino Acid")

    ax1.set_ylabel(label_a, color=color_a)
    ax2.set_ylabel(label_b, color=color_b)

    ax1.set_ylim(0, max(importance_a) * 1.2)
    ax2.set_ylim(0, max(importance_b) * 1.2)

    ax1.set_title(f"Hyperthermophilic ({probability:.2f})")

    ax1.tick_params(axis="y", colors=color_a)
    ax2.tick_params(axis="y", colors=color_b)

    threshold = 0.1

    for bar in bars_a:
        h = bar.get_height()
        if h > threshold:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(importance_a) * 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color_a,
            )

    for bar in bars_b:
        h = bar.get_height()
        if h > threshold:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(importance_b) * 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color_b,
            )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2)

    plt.tight_layout()
    plt.savefig(f"test_importance/{protein[0]}_hyperthermo_compare.png")


def make_importance_general(
    protein: dict,
    data: np.ndarray,  # shape (P, N)
    probability: tuple[float, float],
    labels: list[str],  # length P
    outdir: str = "test_importance",
):
    domain = protein["domain"]
    mutant = protein["mutant"]
    prot_id = protein["prot_id"]

    assert len(domain) == len(mutant), "Domain and mutant must have same length"
    P, N = data.shape
    assert P == len(labels), "labels length must match number of features (P)"

    x = np.arange(N)
    total_width = 0.8
    bar_width = total_width / P

    fig, ax = plt.subplots(figsize=(max(10, N * 0.35), 6))

    fig.suptitle(
        f"Feature importance comparison for protein {prot_id}",
        fontsize=16,
    )

    # Plot bars
    for i in range(P):
        offsets = x - total_width / 2 + i * bar_width + bar_width / 2
        ax.bar(
            offsets,
            data[i],
            width=bar_width,
            label=labels[i],
        )

    # X-axis labels with domain / mutant letters
    xtick_labels = []
    for d, m in zip(domain, mutant):
        if d == m:
            xtick_labels.append(d)
        else:
            xtick_labels.append(f"{d}\n{m}")

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Amino Acid Position")

    ax.set_ylabel("Feature value")

    ax.set_title(
        f"Domain score: {probability[0]:.2f} | Mutant score: {probability[1]:.2f}"
    )

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.legend(ncols=min(P, 5))
    ax.set_ylim(0, max(1.0, np.max(data) * 1.2))

    plt.tight_layout()
    plt.savefig(f"{outdir}/{prot_id}_compare.png")
    plt.close(fig)


def make_importance_diff(
    protein: dict,
    data: np.ndarray,  # shape (P, N)
    probability: tuple[float, float],
    labels: list[str],
    outdir: str = "test_importance",
    same_gap: float = 0.15,
    diff_gap: float = 1.0,
):
    """
    Plot feature importance with compressed unchanged regions.
    """

    domain = protein["domain"]
    mutant = protein["mutant"]
    prot_id = protein["prot_id"]

    assert len(domain) == len(mutant), "Domain and mutant must have same length"

    P, N = data.shape
    assert N == len(domain), "Data column count must match sequence length"
    assert P == len(labels), "labels length must match number of features (P)"

    is_diff = np.array([d != m for d, m in zip(domain, mutant)])

    # Build x positions
    x_positions = [0.0]

    for i in range(1, N):
        prev_diff = is_diff[i - 1]
        curr_diff = is_diff[i]

        # if either side is mutation -> large spacing
        if prev_diff or curr_diff:
            step = diff_gap
        else:
            step = same_gap

        x_positions.append(x_positions[-1] + step)

    x_positions = np.array(x_positions)

    total_width = 0.8
    bar_width = total_width / P

    fig_width = max(10, x_positions[-1] * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    fig.suptitle(
        f"Feature importance comparison for protein {prot_id}",
        fontsize=16,
    )

    # Plot bars only for different residues
    for i in range(P):
        offsets = x_positions[is_diff] - total_width / 2 + i * bar_width + bar_width / 2

        ax.bar(
            offsets,
            data[i, is_diff],
            width=bar_width,
            label=labels[i],
        )

    # Tick labels
    xtick_labels = []

    for d, m in zip(domain, mutant):
        if d == m:
            xtick_labels.append(d)
        else:
            xtick_labels.append(f"{d}\n{m}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=8)

    ax.set_xlabel("Amino Acid Position")
    ax.set_ylabel("Feature value")

    ax.set_title(
        f"Domain score: {probability[0]:.2f} | Mutant score: {probability[1]:.2f}"
    )

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)

    ax.legend(ncols=min(P, 5))
    ax.set_ylim(0, max(1.0, np.max(data) * 1.2))

    plt.tight_layout()
    plt.savefig(f"{outdir}/{prot_id}_compare_diff.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # pass

    # Provided protein
    protein = {
        "prot_id": "A0A1M6DL67",
        "domain": "DRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAG",
        "mutant": "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
    }

    N = len(protein["domain"])
    P = 5

    # Reproducibility
    rng = np.random.default_rng(42)

    # Synthetic feature data: shape (P, N)
    data = rng.uniform(0.0, 1.0, size=(P, N))

    # Optional: emphasize differences where domain != mutant
    for i, (d, m) in enumerate(zip(protein["domain"], protein["mutant"])):
        if d != m:
            data[:, i] += rng.uniform(0.3, 0.7, size=P)

    # Feature labels
    labels = [
        "Hydrophobicity",
        "Charge",
        "Volume",
        "Flexibility",
        "Conservation",
    ]

    # Synthetic probabilities
    probability = (
        float(rng.uniform(0.6, 0.9)),  # domain score
        float(rng.uniform(0.6, 0.9)),  # mutant score
    )
    print(data)

    make_importance_diff(protein, data, probability, labels)
