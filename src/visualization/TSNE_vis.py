import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
python -m src.evaluations.TSNE_vis
"""

FEATURES = 1280

MODE = "basic_1280"
DOMAIN_ATR_PATH = f"training_data/{MODE}/domain_attribution.dat"
MUTANT_ATR_PATH = f"training_data/{MODE}/mutant_attribution.dat"
INPUT_PATH = "datasets/mutants_min:13.71_hev:15.82.json"

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def main():
    with open(INPUT_PATH, "r") as f:
        proteins = json.load(f)

    prot_lengths = [len(p["domain"]) for p in proteins]
    total_residues = sum(prot_lengths)

    # Extract sequences
    print("Extracting sequences...")
    all_sequences = "".join(p["domain"] for p in proteins)

    # Convert amino acids → indices
    aa_indices = np.array([aa_to_idx.get(aa, -1) for aa in all_sequences])

    domain_atr = np.memmap(
        DOMAIN_ATR_PATH,
        dtype=np.float16,
        mode="r",
        shape=(total_residues, FEATURES),
    )

    mutant_atr = np.memmap(
        MUTANT_ATR_PATH,
        dtype=np.float16,
        mode="r",
        shape=(total_residues, FEATURES),
    )

    print(f"Total residues: {total_residues}")

    # Subsampling
    MAX_POINTS = 10000
    if total_residues > MAX_POINTS:
        print(f"Subsampling to {MAX_POINTS} residues...")
        indices = np.random.choice(total_residues, MAX_POINTS, replace=False)
        indices.sort()

        d_subset = domain_atr[indices].astype(np.float32)
        m_subset = mutant_atr[indices].astype(np.float32)
        aa_subset = aa_indices[indices]
    else:
        d_subset = domain_atr[:].astype(np.float32)
        m_subset = mutant_atr[:].astype(np.float32)
        aa_subset = aa_indices

    print("Concatenating...")
    combined_atr = np.concatenate([d_subset, m_subset], axis=0)

    print("Applying TSNE...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    reduced_atr = tsne.fit_transform(combined_atr)

    n_samples = len(d_subset)
    domain_reduced = reduced_atr[:n_samples]
    mutant_reduced = reduced_atr[n_samples:]

    # Visualization
    plt.figure(figsize=(14, 10), dpi=300)

    cmap = plt.get_cmap("tab20")

    # Plot domain
    sc1 = plt.scatter(
        domain_reduced[:, 0],
        domain_reduced[:, 1],
        c=aa_subset,
        cmap=cmap,
        marker="o",
        alpha=0.6,
        s=5,
        label="Domain",
    )

    # Plot mutant
    sc2 = plt.scatter(
        mutant_reduced[:, 0],
        mutant_reduced[:, 1],
        c=aa_subset,
        cmap=cmap,
        marker="s",
        alpha=0.6,
        s=5,
        label="Mutant",
    )

    # Create colorbar with amino acid labels
    cbar = plt.colorbar(sc1, ticks=range(len(AMINO_ACIDS)))
    cbar.ax.set_yticklabels(AMINO_ACIDS)
    cbar.set_label("Amino Acid")

    plt.title(f"TSNE colored by Amino Acid ({MODE})")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.legend(loc="upper right")

    output_plot = "TSNE_amino_acid_colored.png"
    plt.savefig(output_plot, bbox_inches="tight")
    print(f"Plot saved to {output_plot}")


if __name__ == "__main__":
    main()
