import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def make_importance(protein: str, importance: list[float]):
    x = range(len(protein))

    plt.figure(figsize=(8, 5))
    plt.bar(x, importance, color="skyblue")

    # Set x-ticks to the amino acids
    plt.xticks(x, list(protein))

    plt.xlabel("Amino Acid")
    plt.ylabel("Importance")
    plt.ylim([0, (max(importance) * 1.2)])
    plt.title("Amino Acid Importance in Protein Thermostability")

    # Add values on top of bars
    for i, val in enumerate(importance):
        plt.text(i, val + 0.01, f"{val:.2f}", ha="center")

    plt.savefig("test_importance.png")  # save figure to file


if __name__ == "__main__":
    protein = "SMRLOOP"
    importance = [0.1, 0.1, 0.2, 0.3, 0.05, 0.15, 0.1]
