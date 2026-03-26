import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
to run:
python -m src.evaluations.TSNE_vis
"""

FEATURES = 1280  # for one class
# FEATURES = 1280 * 4  # for all classes

MODE = "basic_1280"
DOMAIN_EMB_PATH = f"training_data/{MODE}/domain_embedding.dat"
MUTANT_EMB_PATH = f"training_data/{MODE}/mutant_embedding.dat"
INPUT_PATH = "datasets/mutants_min:13.71_hev:15.82.json"


def main():
    with open(INPUT_PATH, "r") as f:
        proteins = json.load(f)

    prot_lengths = [len(p["domain"]) for p in proteins]
    total_residues = sum(prot_lengths)
    residue_prot_lengths = np.repeat(prot_lengths, prot_lengths)

    # Identify first residues of each protein
    is_first_residue = np.zeros(total_residues, dtype=bool)
    curr = 0
    for l in prot_lengths:
        is_first_residue[curr] = True
        curr += l

    domain_emb = np.memmap(
        DOMAIN_EMB_PATH,
        dtype=np.float16,
        mode="r",
        shape=(total_residues, FEATURES),
    )

    mutant_emb = np.memmap(
        MUTANT_EMB_PATH,
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
        d_subset = domain_emb[indices].astype(np.float32)
        m_subset = mutant_emb[indices].astype(np.float32)
        colors_subset = residue_prot_lengths[indices]
        first_res_subset = is_first_residue[indices]
    else:
        d_subset = domain_emb[:].astype(np.float32)
        m_subset = mutant_emb[:].astype(np.float32)
        colors_subset = residue_prot_lengths
        first_res_subset = is_first_residue

    print("Concatenating...")
    combined_emb = np.concatenate([d_subset, m_subset], axis=0)

    print(f"Applying TSNE directly to {combined_emb.shape[1]} dimensions...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    reduced_emb = tsne.fit_transform(combined_emb)

    n_samples = len(d_subset)
    domain_reduced = reduced_emb[:n_samples]
    mutant_reduced = reduced_emb[n_samples:]

    # Visualization
    plt.figure(figsize=(14, 10), dpi=300)

    # Masks for normal vs first residues
    norm_mask = ~first_res_subset
    first_mask = first_res_subset

    # Plot normal residues (circles)
    sc = plt.scatter(
        domain_reduced[norm_mask, 0],
        domain_reduced[norm_mask, 1],
        c=colors_subset[norm_mask],
        marker="o",
        label="Domain (Residue)",
        alpha=0.5,
        s=3,
        cmap="viridis",
    )

    # Plot normal mutants (squares)
    plt.scatter(
        mutant_reduced[norm_mask, 0],
        mutant_reduced[norm_mask, 1],
        c=colors_subset[norm_mask],
        marker="s",
        label="Mutant (Residue)",
        alpha=0.5,
        s=3,
        cmap="viridis",
    )

    # Plot first residues (red crosses)
    plt.scatter(
        domain_reduced[first_mask, 0],
        domain_reduced[first_mask, 1],
        color="red",
        marker="x",
        label="Protein Start (Domain)",
        s=30,
        linewidths=1,
    )
    plt.scatter(
        mutant_reduced[first_mask, 0],
        mutant_reduced[first_mask, 1],
        color="darkred",
        marker="x",
        label="Protein Start (Mutant)",
        s=30,
        linewidths=1,
    )

    plt.colorbar(sc, label="Protein Length")
    plt.title(f"TSNE of Domain and Mutant Embeddings ({MODE})")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.legend(markerscale=2, loc="upper right")

    output_plot = f"TSNE_{MODE}.png"
    plt.savefig(output_plot, bbox_inches="tight")
    print(f"Plot saved to {output_plot}")


if __name__ == "__main__":
    main()
