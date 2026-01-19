from captum.attr import IntegratedGradients
from src.predictor import Classificator
import numpy as np
import torch


def get_captum_embedding(mdl: Classificator, inp: str) -> np.ndarray:
    """
    Return captum embedding
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ig = IntegratedGradients(mdl.model.forward_embedding)

    _, _, tokens = mdl.batch_converter([("", inp)])
    embedding = mdl.model.embedding(tokens.to(DEVICE))
    attr, _ = ig.attribute(
        embedding,
        target=3,
        return_convergence_delta=True,
        internal_batch_size=50,
        # n_steps=10,
    )
    return attr.squeeze(dim=0).detach().numpy()[1:-1]
