from captum.attr import IntegratedGradients
from src.predictor import Classificator
import numpy as np
import torch


def get_captum_embedding_all_classes(mdl: Classificator, inp: str) -> np.ndarray:
    """
    Return captum embedding for all classes concatenated
    1280 * 4
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ig = IntegratedGradients(mdl.model.forward_embedding)

    _, _, tokens = mdl.batch_converter([("", inp)])
    embedding = mdl.model.embedding(tokens.to(DEVICE))

    attrs_per_class = []

    for target in (0, 1, 2, 3):
        attr, _ = ig.attribute(
            embedding,
            target=target,
            return_convergence_delta=True,
            internal_batch_size=12,
        )

        tensor = attr.squeeze(dim=0).detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # remove special tokens
        attrs_per_class.append(tensor.numpy()[1:-1])

    # concatenate on the feature axis
    return np.concatenate(attrs_per_class, axis=1)


def get_captum_embedding(mdl: Classificator, inp: str) -> np.ndarray:
    """
    Return captum embedding
    1280
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("aaa")
    ig = IntegratedGradients(mdl.model.forward_embedding)

    print("bbb")

    _, _, tokens = mdl.batch_converter([("", inp)])
    embedding = mdl.model.embedding(tokens.to(DEVICE))
    attr, _ = ig.attribute(
        embedding,
        target=3,
        return_convergence_delta=True,
        internal_batch_size=12,
        # n_steps=10,
    )
    tensor = attr.squeeze(dim=0).detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()[1:-1]


if __name__ == "__main__":  # only for testing
    classificator = Classificator()

    result = get_captum_embedding(classificator, "RSGLYAPPNWEYGSTMVVPP")
    print(result.shape)
