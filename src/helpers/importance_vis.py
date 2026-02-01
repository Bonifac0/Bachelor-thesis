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
    probability: float,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Importance for protein {protein[0]}", fontsize=16)

    x = range(len(protein[1]))

    ax.bar(x, importance)
    ax.set_xticks(x)
    ax.set_xticklabels(list(protein[1]))
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Importance")
    ax.set_ylim((0, max(importance) * 1.2))
    ax.set_title(f"Importance for hyperthermophilic ({probability:.2f})")

    for i, val in enumerate(importance):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"test_importance/{protein[0]}_hyperthermo.png")


def make_importance_hyperthermo_compare(
    protein: tuple[str, str],
    importance_a: list[float],
    importance_b: list[float],
    probability: float,
    label_a: str = "Importance",
    label_b: str = "Real decrease",
):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    fig.suptitle(f"Importance comparison for protein {protein[0]}", fontsize=16)

    x = range(len(protein[1]))
    width = 0.4

    color_a = "tab:blue"
    color_b = "tab:orange"

    bars_a = ax1.bar(
        [i - width / 2 for i in x],
        importance_a,
        width=width,
        label=label_a,
        color=color_a,
    )

    bars_b = ax2.bar(
        [i + width / 2 for i in x],
        importance_b,
        width=width,
        label=label_b,
        color=color_b,
    )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(list(protein[1]))
    ax1.set_xlabel("Amino Acid")

    ax1.set_ylabel(label_a, color=color_a)
    ax2.set_ylabel(label_b, color=color_b)

    ax1.set_ylim(0, max(importance_a) * 1.2)
    ax2.set_ylim(0, max(importance_b) * 1.2)

    ax1.set_title(f"Hyperthermophilic ({probability:.2f})")

    ax1.tick_params(axis="y", colors=color_a)
    ax2.tick_params(axis="y", colors=color_b)

    threshold = 0.1

    for bar in bars_a:
        h = bar.get_height()
        if h > threshold:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(importance_a) * 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color_a,
            )

    for bar in bars_b:
        h = bar.get_height()
        if h > threshold:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(importance_b) * 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color_b,
            )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2)

    plt.tight_layout()
    plt.savefig(f"test_importance/{protein[0]}_hyperthermo_compare.png")


if __name__ == "__main__":
    pass
