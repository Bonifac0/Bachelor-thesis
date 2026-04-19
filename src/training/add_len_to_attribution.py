import json
import numpy as np
from tqdm import tqdm

"""
python -m src.training.add_len_to_attribution
"""

FEATURES = 1280  # for one class

MODE = "basic_1280"
DOMAIN_ATR_PATH = f"training_data/{MODE}/domain_attribution.dat"
MUTANT_ATR_PATH = f"training_data/{MODE}/mutant_attribution.dat"

DOMAIN_OUT_PATH = f"training_data/{MODE}_with_len/domain_attribution.dat"
MUTANT_OUT_PATH = f"training_data/{MODE}_with_len/mutant_attribution.dat"


INPUT_PATH = "datasets/mutants_min:13.71_hev:15.82.json"


def main():
    with open(INPUT_PATH, "r") as f:
        proteins = json.load(f)

    prot_lengths = [len(p["domain"]) for p in proteins]
    total_residues = sum(prot_lengths)
    residue_prot_lengths = np.repeat(prot_lengths, prot_lengths).astype(np.float16)

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

    NEW_FEATURES = FEATURES + 1
    print(f"Creating new attributions with shape ({total_residues}, {NEW_FEATURES})...")

    domain_atr_out = np.memmap(
        DOMAIN_OUT_PATH,
        dtype=np.float16,
        mode="w+",
        shape=(total_residues, NEW_FEATURES),
    )

    mutant_atr_out = np.memmap(
        MUTANT_OUT_PATH,
        dtype=np.float16,
        mode="w+",
        shape=(total_residues, NEW_FEATURES),
    )

    CHUNK_SIZE = 100000
    for start in tqdm(range(0, total_residues, CHUNK_SIZE)):
        end = min(start + CHUNK_SIZE, total_residues)

        len_chunk = residue_prot_lengths[start:end].reshape(-1, 1)

        # Domain: Copy 1280 features and append length
        domain_atr_out[start:end, :FEATURES] = domain_atr[start:end]
        domain_atr_out[start:end, FEATURES:] = len_chunk

        # Mutant: Copy 1280 features and append length
        mutant_atr_out[start:end, :FEATURES] = mutant_atr[start:end]
        mutant_atr_out[start:end, FEATURES:] = len_chunk

    # Flush to ensure data is written to disk
    domain_atr_out.flush()
    mutant_atr_out.flush()
    print(f"Done. Saved to {DOMAIN_OUT_PATH} and {MUTANT_OUT_PATH}")


if __name__ == "__main__":
    main()
