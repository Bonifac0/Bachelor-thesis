from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """
    Compute the Levenshtein distance between two sequences (strings, lists, etc.)
    using a memory-efficient dynamic programming algorithm.
    """

    # Ensure a is the shorter one to reduce memory usage
    if len(a) > len(b):
        a, b = b, a

    len_a = len(a)
    len_b = len(b)

    # If one is empty, distance is length of the other
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    # Previous and current row of DP table
    previous_row = list(range(len_a + 1))
    current_row = [0] * (len_a + 1)

    for j in range(1, len_b + 1):
        bj = b[j - 1]
        current_row[0] = j  # cost of transforming empty a -> b[:j]

        # Compute current_row[i] from previous_row and current_row[i-1]
        for i in range(1, len_a + 1):
            cost = 0 if a[i - 1] == bj else 1

            insertion = current_row[i - 1] + 1  # a[:i] to b[:j-1] + insert
            deletion = previous_row[i] + 1  # a[:i-1] to b[:j] + delete
            substitution = previous_row[i - 1] + cost

            # Take the cheapest operation
            current_row[i] = min(insertion, deletion, substitution)

        # Swap rows instead of copying
        previous_row, current_row = current_row, previous_row

    return previous_row[len_a]


if __name__ == "__main__":
    s1 = "LMVIGAGHDAIPICSVAASLGWYVYLWDERIHTSHYSEYQGAAVVDNHKSDKQKNYSLLAGFNAVILKSHNLSIDAFWLSQIEQYQQQIAYIGLLGPKYRKGKVIEAAELQHAGWAQERIFSPAGLEIGGDTAEAIALSILSQVMLATLIKKTGSSYRKEGAMMFIAPSGETIGALSGGCLEKDIVHQAHRLSFSDSSTVIEYD"
    s2 = "LVVVGFGEVARRVSEVAVAAGFNVAALGHSAQGAVYRGELTDLEKLVSEGSAVVVANEGGHPSDVDVVETALRRGAGYVALLASQRRAALVVKELLKRGIPREAIAERLRSPAGLDIGAKTAGEIAVSIVAEV"
    print(levenshtein(s1, s2))  # 3
    print(min(len(s1), len(s2)))
