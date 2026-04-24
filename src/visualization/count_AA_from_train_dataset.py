import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

"""
python -m src.visualization.count_AA_from_train_dataset
"""


def load_sequence(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def split_by_mask(sequence, mask):
    seq0 = []
    seq1 = []

    for ch, m in zip(sequence, mask):
        if m == 0:
            seq0.append(ch)
        else:
            seq1.append(ch)

    return "".join(seq0), "".join(seq1)


def count_chars(seq):
    return Counter(seq)


def plot_comparison(counter0, counter1, output_pdf):
    keys = sorted(set(counter0.keys()) | set(counter1.keys()))

    vals0 = [counter0.get(k, 0) for k in keys]
    vals1 = [counter1.get(k, 0) for k in keys]

    x = np.arange(len(keys))
    width = 0.4

    plt.figure(figsize=(10, 5))

    plt.bar(x - width / 2, vals0, width)
    plt.bar(x + width / 2, vals1, width)

    plt.xticks(x, keys)
    plt.xlabel("Residue")
    plt.ylabel("Count")
    plt.title("Residue distribution comparison")

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()


def main(seq_path, mask_path, output_pdf="comparison.pdf"):
    sequence = load_sequence(seq_path)

    mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    mask = np.asarray(mask)

    if len(sequence) != len(mask):
        raise ValueError("Sequence and mask length mismatch")

    seq0, seq1 = split_by_mask(sequence, mask)

    c0 = count_chars(seq0)
    c1 = count_chars(seq1)

    plot_comparison(c0, c1, output_pdf)


if __name__ == "__main__":
    main(
        "training_data/basic_1280/amino_acids.txt",
        "training_data/basic_1280/y.dat",
        "comparison.pdf",
    )
