import torch
from src.helpers.captum_embedding import get_captum_embedding
from src.training.model_definitions import ImportancePredictor
from src.predictor import Classificator
from src.helpers.importance_vis import make_importance_hyperthermo
import numpy as np


"""
to run:
python -m src.training.run_model
"""


class ModelRunner:
    def __init__(
        self,
        classificator: Classificator,
        model_path: str = "resources/importance_model.pt",
    ):
        self.classificator = classificator
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImportancePredictor()
        self.model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        self.model.to(self.DEVICE)
        self.model.eval()

    def predict_importance(self, seq) -> np.ndarray:
        # Compute embeddings
        emb = get_captum_embedding(self.classificator, seq)  # (N, 1280)

        # Convert to torch tensor
        x = torch.from_numpy(emb).float().to(self.DEVICE)

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)  # shape: (N,)
            probs = torch.sigmoid(logits)  # convert to [0,1]

        return probs.cpu().numpy()


if __name__ == "__main__":
    classificator = Classificator()
    runner = ModelRunner(classificator, "importance_model.pt")

    proteins = [("pokus", "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG")]
    cold_shock = [  # 18 GB gpu memory
        (  # 270MB/aminoacid
            "term",
            "MQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKL-",
        ),
        (
            "mezo",
            "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA",
        ),
    ]
    pokus = [
        (
            "domain",
            "DRDGLYAPANWEPGSTMVVPPTMSDEEAETGFAG",
        ),
        (
            "mutant",
            "MRSGLYAPPNWEYGSTMVVPPTMSSEEAETGGAG",
        ),
    ]
    for protein in pokus:
        importance_scores = runner.predict_importance(protein[1])

        print("Residue\tImportance")
        for i, score in enumerate(importance_scores):
            print(f"{protein[1][i]}\t{score:.4f}")

        probability = classificator.classify([protein])
        print(probability[0])
        make_importance_hyperthermo(protein, importance_scores, probability[0][3])
