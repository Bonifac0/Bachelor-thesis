import torch
import torch.nn as nn
from src.training.captum_for_training import get_captum_embedding

# from src.training.train_predictor import ImportanceModel
from src.predictor import Classificator
from src.heplers.importance_vis import make_importance_hyperthermo
import numpy as np


"""
to run:
python -m src.training.run_model

because pyhon need to load packages
"""

# =========================
# Configuration
# =========================

MODEL_PATH = "importance_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES = 1280


# =========================
# Load model
# =========================


class ImportanceModel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        self.norm = nn.LayerNorm(FEATURES)
        self.linear = nn.Linear(FEATURES, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.norm(x)
        return self.linear(x).squeeze(-1)


model = ImportanceModel(mean=np.zeros(FEATURES), std=np.ones(FEATURES))
model.load_state_dict(torch.load("importance_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()


# =========================
# Prediction function
# =========================


def predict_importance(mdl: Classificator, seq) -> np.ndarray:
    # Compute embeddings
    emb = get_captum_embedding(mdl, seq)  # (N, 1280)

    # Convert to torch tensor
    x = torch.from_numpy(emb).float().to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(x)  # shape: (N,)
        probs = torch.sigmoid(logits)  # convert to [0,1]

    return probs.cpu().numpy()


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    classificator = Classificator()

    protein = ("pokus", "MKTFFVAGV")

    importance_scores = predict_importance(classificator, protein[1])

    print("Residue\tImportance")
    for i, score in enumerate(importance_scores):
        print(f"{protein[1][i]}\t{score:.4f}")

    probability = classificator.classify([protein])
    print(probability[0])
    make_importance_hyperthermo(protein, importance_scores, probability[0])
