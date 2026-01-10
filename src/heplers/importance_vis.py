import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


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
    probability: list[float],
):
    # Select only the 4th element (hyperthermophilic)
    idx = 3

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Importance for protein {protein[0]}", fontsize=16)

    x = range(len(protein[1]))

    ax.bar(x, importance)
    ax.set_xticks(x)
    ax.set_xticklabels(list(protein[1]))
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Importance")
    ax.set_ylim((0, max(importance) * 1.2))
    ax.set_title(f"Importance for hyperthermophilic ({probability[idx]:.2f})")

    for i, val in enumerate(importance):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"test_importance/{protein[0]}_hyperthermo.png")


if __name__ == "__main__":
    pass
