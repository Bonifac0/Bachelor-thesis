import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def make_importance(
    protein: str, importance: list[list[float]], probability: list[float]
):
    labels = ["psychrophilic", "mesophilic", "thermophilic", "hyperthermophilic"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, imp in enumerate(importance):
        ax = axes[idx]
        x = range(len(protein))

        ax.bar(x, imp)
        ax.set_xticks(x)
        ax.set_xticklabels(list(protein))
        ax.set_xlabel("Amino Acid")
        ax.set_ylabel("Importance")
        ax.set_ylim([0, max(imp) * 1.2])
        ax.set_title(f"Importance for {labels[idx]} ({probability[idx]})")

        for i, val in enumerate(imp):
            ax.text(i, val + 0.01, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig("test_importance.png")


if __name__ == "__main__":
    pass
