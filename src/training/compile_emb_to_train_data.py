import json
import numpy as np

"""
to run:
python -m src.training.compile_emb_to_train_data
"""

# FEATURES = 1280  # for one class
FEATURES = 1280 * 4  # for all classes

MODE = "combine_1280+4"
DOMAIN_EMB_PATH = f"training_data/{MODE}/domain_embedding.dat"
MUTANT_EMB_PATH = f"training_data/{MODE}/mutant_embedding.dat"
INPUT_PATH = "datasets/mutants_min:13.71_hev:15.82.json"

OUT_X_PATH = f"training_data/{MODE}/X.dat"
OUT_Y_PATH = f"training_data/{MODE}/y.dat"


def compute_difference_mask(domain: str, mutant: str) -> np.ndarray:
    assert len(domain) == len(mutant)
    return np.fromiter(
        (d != m for d, m in zip(domain, mutant)),
        dtype=np.bool_,
        count=len(domain),
    )


def main():
    with open(INPUT_PATH, "r") as f:
        proteins = json.load(f)

    total_residues = sum(len(p["domain"]) for p in proteins)

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

    # count total differing residues
    diff_count = sum(
        np.count_nonzero(compute_difference_mask(p["domain"], p["mutant"]))
        for p in proteins
    )

    total_samples = diff_count * 2

    X = np.memmap(
        OUT_X_PATH,
        dtype=np.float16,
        mode="w+",
        shape=(total_samples, FEATURES),
    )

    y = np.memmap(
        OUT_Y_PATH,
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples,),
    )

    emb_idx = 0
    out_idx = 0

    for p in proteins:
        domain_seq = p["domain"]
        mutant_seq = p["mutant"]

        mask = compute_difference_mask(domain_seq, mutant_seq)

        for i in range(len(domain_seq)):
            if not mask[i]:
                emb_idx += 1
                continue

            # domain embedding
            X[out_idx] = domain_emb[emb_idx]
            y[out_idx] = 0
            out_idx += 1

            # mutant embedding
            X[out_idx] = mutant_emb[emb_idx]
            y[out_idx] = 1
            out_idx += 1

            emb_idx += 1

    X.flush()
    y.flush()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
