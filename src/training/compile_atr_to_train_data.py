import json
import numpy as np

"""
python -m src.training.compile_atr_to_train_data
"""

# FEATURES = 1280  # for one class
# FEATURES = 1280 * 4  # for all classes
FEATURES = 1281

MODE = "basic_1280_with_len"
DOMAIN_ATB_PATH = f"training_data/{MODE}/domain_attribution.dat"
MUTANT_ATB_PATH = f"training_data/{MODE}/mutant_attribution.dat"
INPUT_PATH = "datasets/mutants_min:13.71_hev:15.82.json"

OUT_X_PATH = f"training_data/{MODE}/X.dat"
OUT_Y_PATH = f"training_data/{MODE}/y.dat"
OUT_LENGTHS_PATH = f"training_data/{MODE}/lengths.dat"
AA_OUT_PATH = f"training_data/{MODE}/amino_acids.txt"


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

    domain_atb = np.memmap(
        DOMAIN_ATB_PATH,
        dtype=np.float16,
        mode="r",
        shape=(total_residues, FEATURES),
    )

    mutant_atb = np.memmap(
        MUTANT_ATB_PATH,
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

    lengths = np.memmap(
        OUT_LENGTHS_PATH,
        dtype=np.uint16,
        mode="w+",
        shape=(total_samples,),
    )

    atb_idx = 0
    out_idx = 0

    aa_out = []

    for p in proteins:
        domain_seq = p["domain"]
        mutant_seq = p["mutant"]

        mask = compute_difference_mask(domain_seq, mutant_seq)

        for i in range(len(domain_seq)):
            if not mask[i]:
                atb_idx += 1
                continue

            # domain attribution
            X[out_idx] = domain_atb[atb_idx]
            y[out_idx] = 0
            lengths[out_idx] = len(domain_seq)
            aa_out.append(domain_seq[i])
            out_idx += 1

            # mutant attribution
            X[out_idx] = mutant_atb[atb_idx]
            y[out_idx] = 1
            lengths[out_idx] = len(domain_seq)
            aa_out.append(mutant_seq[i])
            out_idx += 1

            atb_idx += 1

    X.flush()
    y.flush()
    lengths.flush()

    with open(AA_OUT_PATH, "w") as f:
        f.write("".join(aa_out))

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"lengths shape: {lengths.shape}")


if __name__ == "__main__":
    main()
