import torch
from src.helpers.captum_embedding import get_captum_embedding
from src.predictor import Classificator
from src.helpers.importance_vis import make_importance_hyperthermo
import numpy as np
from src.training.model_definitions import (
    ImportancePredictorWithLengthAndHL,
    ImportancePredictorWithLength,
    ImportancePredictorWithHL,
    ImportancePredictorWithNormalization,
    ImportancePredictorBasic,
    ImportancePredictorWith2HL,
    ImportancePredictorAllClassWithHL,
)


"""
python -m src.training.run_model

Contain ModelRunner class used for running importance predictor
"""


class ModelRunner:
    """
    Wrapper for initializing an importance prediction model.

    The `ARCHITECTURE` argument selects the model type and corresponding
    `.pt` file in the models directory.

    Supported values:
        - "basic" → ImportancePredictorBasic
        - "len_and_HL_16" → ImportancePredictorWithLengthAndHL (HL=16)
        - "length" → ImportancePredictorWithLength
        - "normalization" → ImportancePredictorWithNormalization
        - "HL_16" → ImportancePredictorWithHL (HL=16)
        - "2HL_64_16" → ImportancePredictorWith2HL (64, 16)
        - "all_class_HL_16" → ImportancePredictorAllClassWithHL (HL=16)

    The `Classificator` instance is accasable from outside.
    """

    def __init__(self, ARCHITECTURE: str):
        match ARCHITECTURE:
            case "basic":
                self.model = ImportancePredictorBasic()
            case "len_and_HL_16":
                self.model = ImportancePredictorWithLengthAndHL(16)
            case "length":
                self.model = ImportancePredictorWithLength()
            case "normalization":
                self.model = ImportancePredictorWithNormalization()
            case "HL_16":
                self.model = ImportancePredictorWithHL(16)
            case "2HL_64_16":
                self.model = ImportancePredictorWith2HL(64, 16)
            case "all_class_HL_16":
                self.model = ImportancePredictorAllClassWithHL(16)
            case _:
                raise ValueError(f"Invalid ARCHITECTURE: {ARCHITECTURE}")

        self.classificator = Classificator()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint (model + normalization)
        checkpoint = torch.load(f"models/{ARCHITECTURE}.pt", map_location=self.DEVICE)

        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.DEVICE)
        self.model.eval()

        # Load normalization stats
        norm = checkpoint["normalization"]
        self.mean_emb = norm["mean_emb"].to(self.DEVICE)
        self.std_emb = norm["std_emb"].to(self.DEVICE)
        if self.model.USE_LENGTH:
            self.mean_len = norm["mean_len"].to(self.DEVICE)
            self.std_len = norm["std_len"].to(self.DEVICE)

    def normalize_input_with_len(
        self, x: torch.Tensor, length_feature: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize embeddings and length feature separately.
        x: (N, 1280)
        length_feature: (N, 1)
        """
        emb = (x - self.mean_emb) / self.std_emb
        length = (length_feature - self.mean_len) / self.std_len
        return torch.cat([emb, length], dim=-1)

    def predict_importance(self, seq) -> np.ndarray:
        # Compute embeddings
        emb = get_captum_embedding(self.classificator, seq)  # (N, 1280)
        x = torch.from_numpy(emb).float().to(self.DEVICE)

        if self.model.USE_LENGTH:
            # Create length feature for each residue
            seq_len = torch.full(
                (x.shape[0], 1), fill_value=len(seq), device=x.device, dtype=x.dtype
            )
            # Normalize embeddings and length feature
            x = self.normalize_input_with_len(x, seq_len)
        else:
            x = (x - self.mean_emb) / self.std_emb

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)  # shape: (N,)
            probs = torch.sigmoid(logits)  # convert to [0,1]

        return probs.cpu().numpy()


if __name__ == "__main__":
    runner = ModelRunner("HL_16")

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

        probability = runner.classificator.classify([protein])
        print(probability[0])
        make_importance_hyperthermo(protein, importance_scores, probability[0][3])
