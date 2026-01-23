import matplotlib.pyplot as plt
import torch
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, average_precision_score
from src.training.model_definitions import ImportancePredictor, DatasetHandler
import numpy as np
import os

"""
to run:
python -m src.training.predictor_tester
"""

# maybe obsolete, TODO try wandb.ia


def model_tester(model: ImportancePredictor, test_loader):
    PRECI_RECALL_CURVE_FILE = "precision_recall_curve.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)

            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    y_scores = expit(all_logits)  # sigmoid
    y_true = all_labels

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    print(f"AUPRC: {auprc:.6f}")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PRECI_RECALL_CURVE_FILE, dpi=300)
    plt.close()


if __name__ == "__main__":  # only for testing
    MODEL_PATH = "importance_model.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEATURES = 1280
    X_PATH = "X.dat"
    Y_PATH = "y.dat"
    TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue
    DATASET_SPLIT = (0.8, 0.1, 0.1)

    model = ImportancePredictor(mean=np.zeros(FEATURES), std=np.ones(FEATURES))
    model.load_state_dict(torch.load("importance_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    X = np.memmap(X_PATH, dtype=np.float16, mode="r", shape=(TOTAL_RESIDUES, FEATURES))
    y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))
    dataset = DatasetHandler(X, y, DATASET_SPLIT)

    model_tester(model, dataset.test_loader)
