import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

"""
python -m src.visualization.TSNE_vis
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

    print("Extracting sequences...")
    all_sequences = "".join(p["domain"] for p in proteins)

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

    MAX_POINTS = 5000
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

    # duplicate aa labels for combined data
    aa_combined = np.concatenate([aa_subset, aa_subset], axis=0)

    print("Applying TSNE...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    reduced_atr = tsne.fit_transform(combined_atr)

    # Visualization
    plt.figure(figsize=(14, 10))

    cmap = plt.get_cmap("tab20")

    # discrete normalization (fixes colorbar alignment)
    bounds = np.arange(len(AMINO_ACIDS) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    sc = plt.scatter(
        reduced_atr[:, 0],
        reduced_atr[:, 1],
        c=aa_combined,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        s=10,
        rasterized=False,
    )

    # Colorbar
    cbar = plt.colorbar(sc, ticks=np.arange(len(AMINO_ACIDS)))
    cbar.ax.set_yticklabels(AMINO_ACIDS)
    cbar.set_label("Amino Acid", fontsize=18)
    cbar.ax.tick_params(length=0, labelsize=14)

    plt.title("TSNE colored by Amino Acid", fontsize=24)
    plt.xlabel("TSNE Dimension 1", fontsize=18)
    plt.ylabel("TSNE Dimension 2", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("TSNE_amino_acid_colored.pdf", bbox_inches="tight")

    print("Plot saved to TSNE_amino_acid_colored.pdf")


if __name__ == "__main__":
    main()
