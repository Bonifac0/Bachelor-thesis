import matplotlib
import numpy as np
from src.helpers.levenshtein import levenshtein

matplotlib.use("agg")
import matplotlib.pyplot as plt

"""
python -m src.helpers.importance_vis
"""


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
    data: np.ndarray,
    probability: tuple[float, float],
    labels: list[str],
    outdir: str = ".",
):
    """
    Plot importance only for mutated residues.

    Residue numbering is based on the full protein sequence
    using approximate Levenshtein alignment.
    """

    sequence = protein["sequence"]
    domain = protein["domain"]
    mutant = protein["mutant"]
    prot_id = protein["prot_id"]

    assert len(domain) == len(mutant)

    P, N = data.shape

    assert N == len(domain)
    assert P == len(labels)

    # ============================================================
    # Find best domain match inside full sequence
    # ============================================================

    best_start = None
    best_distance = float("inf")

    for start in range(len(sequence) - len(domain) + 1):
        window = sequence[start : start + len(domain)]

        dist = levenshtein(window, domain)

        if dist < best_distance:
            best_distance = dist
            best_start = start

    if best_start is None:
        raise ValueError("Could not map domain into sequence")

    # Full-sequence residue numbering
    residue_numbers = np.arange(
        best_start + 1,
        best_start + len(domain) + 1,
    )

    # ============================================================
    # Keep only mutated residues
    # ============================================================

    diff_idx = [i for i, (d, m) in enumerate(zip(domain, mutant)) if d != m]

    if not diff_idx:
        raise ValueError("No mutations found")

    # Global residue range in full-sequence numbering
    global_start = residue_numbers[diff_idx[0]]
    global_end = residue_numbers[diff_idx[-1]]

    data = data[:, diff_idx]

    # Constant spacing
    x = np.arange(len(diff_idx))

    # Labels with sequence numbering
    xticklabels = [
        f"{residue_numbers[idx]}\n{domain[idx]}→{mutant[idx]}" for idx in diff_idx
    ]

    # ============================================================
    # Plot
    # ============================================================

    total_width = 0.8
    bar_width = total_width / P

    fig, ax = plt.subplots(figsize=(10, 6))

    # ax.margins(x=0)

    fig.suptitle(
        f"Feature importance for protein {prot_id}",
        fontsize=20,
    )

    colors = ["C1", "C1", "C0", "C0"]
    hatches = ["", "//", "", "//"]
    alphas = [1.0, 0.7, 1.0, 0.7]

    for i in range(P):
        offsets = x - total_width / 2 + i * bar_width + bar_width / 2

        ax.bar(
            offsets,
            data[i],
            width=bar_width,
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            alpha=alphas[i],
        )

    # Vertical separators between residues
    for xpos in np.arange(len(diff_idx) - 1) + 0.5:
        ax.axvline(
            xpos,
            color="black",
            linestyle="-",
            linewidth=0.5,
            alpha=0.3,
            zorder=0,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        xticklabels,
        fontsize=12,
    )

    ax.set_xlabel(
        "Amino Acid",
        fontsize=18,
    )

    ax.set_ylabel(
        "Predicted value",
        fontsize=18,
    )

    ax.set_title(
        f"Domain score: {probability[0]:.2f} | "
        f"Mutant score: {probability[1]:.2f} | "
        f"Region: {global_start}-{global_end}",
        fontsize=16,
    )

    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        alpha=0.5,
    )

    ax.legend()

    ax.set_ylim(
        0,
        max(1.0, np.max(data) * 1.2),
    )

    plt.tight_layout()

    plt.savefig(f"{outdir}/{prot_id}_compare.pdf")

    plt.close(fig)


def make_importance_diff_context(
    protein: dict,
    data: np.ndarray,  # shape (P, N)
    probability: tuple[float, float],
    labels: list[str],
    outdir: str = ".",
):
    """
    Plot feature importance for mutated residues while compressing
    long unchanged regions using "...".

    Features:
    - Approximate domain-to-sequence mapping using Levenshtein distance
    - Residue numbering based on full protein sequence
    - Long unchanged regions compressed:
            ABCDEFG -> A ... G
    - Bars plotted ONLY for mutated residues
    """

    sequence: str = protein["sequence"]
    domain: str = protein["domain"]
    mutant: str = protein["mutant"]
    prot_id: str = protein["prot_id"]

    assert len(domain) == len(mutant), "Domain and mutant must have same length"
    P, N = data.shape
    assert N == len(domain), "Data column count must match domain length"
    assert P == len(labels), "labels length must match number of features"

    # ============================================================
    # Find best approximate alignment of domain inside sequence
    # ============================================================

    best_start = None
    best_distance = float("inf")

    for start in range(len(sequence) - len(domain) + 1):
        window = sequence[start : start + len(domain)]

        dist = levenshtein(window, domain)

        if dist < best_distance:
            best_distance = dist
            best_start = start

    if best_start is None:
        raise ValueError("Could not map domain into sequence")

    # Warn if alignment quality is weak
    if best_distance > len(domain) * 0.1:
        print(f"Warning: weak domain match (distance={best_distance})")

    # Residue numbering in coordinates of FULL sequence
    residue_numbers = np.arange(
        best_start + 1,
        best_start + len(domain) + 1,
    )

    # ============================================================
    # Determine mutated positions
    # ============================================================

    is_diff = np.array([d != m for d, m in zip(domain, mutant)])

    # ============================================================
    # Crop to region around mutations only
    # ============================================================

    diff_indices = np.where(is_diff)[0]

    if len(diff_indices) == 0:
        raise ValueError("No differences found between domain and mutant")

    pad = 1  # context around mutation

    start = max(0, diff_indices[0] - pad)
    end = min(N, diff_indices[-1] + pad + 1)

    global_start = residue_numbers[start]
    global_end = residue_numbers[end - 1]

    domain = domain[start:end]
    mutant = mutant[start:end]
    data = data[:, start:end]
    residue_numbers = residue_numbers[start:end]
    is_diff = is_diff[start:end]

    N = end - start

    # ============================================================
    # Compress long unchanged regions
    #
    # Example:
    #     ABCDEFG -> A ... G
    #
    # Rules:
    # - differing residues always shown
    # - unchanged runs >=3 are compressed
    # - edge residues kept for context
    # ============================================================

    visible = np.zeros(N, dtype=bool)
    ellipsis_positions = []
    i = 0

    while i < N:
        # Always keep mutated residues visible
        if is_diff[i]:
            visible[i] = True
            i += 1
            continue

        # Find contiguous unchanged block
        start = i

        while i < N and not is_diff[i]:
            i += 1

        end = i
        length = end - start

        # Compress long unchanged regions
        if length >= 3:
            # Keep edges visible
            visible[start] = True
            visible[end - 1] = True

            # Position where "..." will appear
            ellipsis_positions.append((start + end - 1) // 2)

        else:
            # Keep short regions fully visible
            visible[start:end] = True

    ellipsis_positions = set(ellipsis_positions)

    # ============================================================
    # Build plotted labels and indices
    # ============================================================

    plotted_indices = []
    plotted_labels = []

    for idx in range(N):
        # Visible residue
        if visible[idx]:
            plotted_indices.append(idx)

            pos = residue_numbers[idx]

            d = domain[idx]
            m = mutant[idx]

            # Unchanged residue
            if d == m:
                plotted_labels.append(f"{pos}\n{d}")

            # Mutated residue
            else:
                plotted_labels.append(f"{pos}\n{d}\n{m}")

        # Compressed region marker
        elif idx in ellipsis_positions:
            plotted_indices.append(idx)
            plotted_labels.append("...")

    plotted_indices = np.array(plotted_indices)

    # ============================================================
    # Build adaptive x positions
    #
    # Spacing rules:
    #
    # diff <-> diff         : large spacing
    # diff <-> context      : medium spacing
    # context <-> context   : small spacing
    # ============================================================

    x_positions = [0.0]

    for i in range(1, len(plotted_indices)):
        prev_idx = plotted_indices[i - 1]
        curr_idx = plotted_indices[i]

        prev_is_diff = prev_idx < N and is_diff[prev_idx]
        curr_is_diff = curr_idx < N and is_diff[curr_idx]

        # Two mutated residues
        if prev_is_diff and curr_is_diff:
            step = 0.6

        # Transition between mutation and context
        elif prev_is_diff or curr_is_diff:
            step = 0.35

        # Two context residues / ellipsis
        else:
            step = 0.15

        x_positions.append(x_positions[-1] + step)

    x_positions = np.array(x_positions)

    total_width = 0.45  # was 0.8
    bar_width = total_width / P

    # Plot bars ONLY for mutated residues
    bar_mask = np.array([idx < N and is_diff[idx] for idx in plotted_indices])

    bar_indices = plotted_indices[bar_mask]

    # ============================================================
    # Figure setup
    # ============================================================

    fig_width = max(6, len(x_positions) * 0.25)

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.margins(x=0)

    fig.suptitle(
        f"Feature importance comparison for protein {prot_id}",
        fontsize=24,
    )

    # ============================================================
    # Visual styling
    # ============================================================

    # Orange pair + blue pair
    colors = ["C1", "C1", "C0", "C0"]

    # First solid, second hatched
    hatches = ["", "//", "", "//"]

    # Slight transparency for hatched bars
    alphas = [1.0, 0.7, 1.0, 0.7]

    # ============================================================
    # Plot bars
    # ============================================================

    for i in range(P):
        offsets = (
            x_positions[bar_mask] - total_width / 2 + i * bar_width + bar_width / 2
        )

        ax.bar(
            offsets,
            data[i, bar_indices],
            width=bar_width,
            label=labels[i],
            color=colors[i],
            hatch=hatches[i],
            alpha=alphas[i],
        )

    # ============================================================
    # Axis labels and formatting
    # ============================================================

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        plotted_labels,
        fontsize=8,
    )

    ax.set_xlabel(
        "Amino Acid",
        fontsize=18,
    )

    ax.set_ylabel(
        "Predicted value",
        fontsize=18,
    )

    ax.set_title(
        f"Domain score: {probability[0]:.2f} | "
        f"Mutant score: {probability[1]:.2f} | "
        f"Region: {global_start}-{global_end}",
        fontsize=16,
    )

    # Reference threshold line
    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        alpha=0.5,
    )

    ax.legend(ncols=min(P, 5))

    ax.set_ylim(
        0,
        max(1.0, np.max(data) * 1.2),
    )

    plt.tight_layout()

    plt.savefig(f"{outdir}/{prot_id}_compare.pdf")

    plt.close(fig)


if __name__ == "__main__":
    # pass

    # Provided protein
    protein = {
        "prot_id": "pokusny",
        "sequence": "AASADRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAGACVEVERGBE",
        "domain": "SADRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAG",
        "mutant": "SAMWSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
    }

    N = len(protein["domain"])
    P = 4

    # Reproducibility
    rng = np.random.default_rng(42)

    # Synthetic feature data: shape (P, N)
    data = rng.uniform(0.0, 1.0, size=(P, N))

    # Feature labels
    labels = [
        "Hydrophobicity",
        "Charge",
        "Volume",
        "Flexibility",
    ]

    # Synthetic probabilities
    probability = (
        float(rng.uniform(0.6, 0.9)),  # domain score
        float(rng.uniform(0.6, 0.9)),  # mutant score
    )

    make_importance_diff(protein, data, probability, labels)
