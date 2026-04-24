import torch
from src.helpers.captum_attribution import get_captum_attribution
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
    ImportancePredictorWith3HL,
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
        - "3HL_64_32_16" → ImportancePredictorWith3HL (64, 32, 16)
        - "all_class_HL_16" → ImportancePredictorAllClassWithHL (HL=16)

    The `Classificator` instance can be skipped if attributions will be provided.
    """

    def __init__(self, ARCHITECTURE: str, require_classificator=True):
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
            case "3HL_64_32_16":
                self.model = ImportancePredictorWith3HL(64, 32, 16)
            case "all_class_HL_16":
                self.model = ImportancePredictorAllClassWithHL(16)
            case _:
                raise ValueError(f"Invalid ARCHITECTURE: {ARCHITECTURE}")

        if require_classificator:
            self.classificator = Classificator()
        else:
            self.classificator = None

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint (model + normalization)
        checkpoint = torch.load(f"models/{ARCHITECTURE}.pt", map_location=self.DEVICE)

        # Load model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.DEVICE)
        self.model.eval()

        # Load normalization stats
        norm = checkpoint["normalization"]
        if "mean_atr" in norm:
            self.mean_atr = norm["mean_atr"].to(self.DEVICE)
            self.std_atr = norm["std_atr"].to(self.DEVICE)
        else:  # legacy reasons
            self.mean_atr = norm["mean_emb"].to(self.DEVICE)
            self.std_atr = norm["std_emb"].to(self.DEVICE)

        if self.model.USE_LENGTH:
            self.mean_len = norm["mean_len"].to(self.DEVICE)
            self.std_len = norm["std_len"].to(self.DEVICE)

    def normalize_input(self, x: torch.Tensor, seq_len: int = 0) -> torch.Tensor:
        """
        Normalize attributions and length feature separately.
        If model support length feature, provide it in `seq_len`.
        """
        if self.model.USE_LENGTH:
            assert seq_len != 0, "length of sequence can not be 0"
            # Create length feature for each residue
            length_feature = torch.full(
                (x.shape[0], 1), fill_value=seq_len, device=x.device, dtype=x.dtype
            )
            # Normalize attributions and length feature
            atr = (x - self.mean_atr) / self.std_atr
            length = (length_feature - self.mean_len) / self.std_len
            return torch.cat([atr, length], dim=-1)
        else:
            return (x - self.mean_atr) / self.std_atr

    def predict_importance(self, seq: str, atr: np.ndarray | None = None) -> np.ndarray:
        """
        If you have atribution vector already, you can pass it and save a lot time.
        The atr should originate from get_captum_attribution function.
        """
        if atr is None:
            assert self.classificator is not None, "Classificator innit skipped"
            atr = get_captum_attribution(self.classificator, seq)
        x = torch.from_numpy(atr).float().to(self.DEVICE)

        x = self.normalize_input(x, len(seq))
        # Forward pass
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy()


if __name__ == "__main__":
    runner = ModelRunner("2HL_64_16")

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
