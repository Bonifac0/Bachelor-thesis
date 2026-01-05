from captum.attr import IntegratedGradients
from src.predictor import Classificator
import torch

"TODO maybe finish if necesary"

"""
to run:
python -m src.captum.captum_memory_probe

because pyhon need to load Clasificator
"""


def captum_memory_probe(mdl: Classificator, inp: tuple[str, str]):
    ig = IntegratedGradients(mdl.model.forward_embedding)
    _, _, tokens = mdl.batch_converter([inp])  # can not be done all at once
    embedding = mdl.model.embedding(tokens.to(DEVICE))

    attr_baseline = ig.attribute(
        embedding,
        target=3,  # hyperthermophilic
        internal_batch_size=40,
        n_steps=200,
    ).squeeze(0)  # get rid of first dimention (remove if multiple proteins)

    # deviations = []

    # for i in [50]:  # , 4, 8, 16]:
    #     attr = ig.attribute(c
    #         embedding,
    #         target=3,  # hyperthermophilic
    #         internal_batch_size=i,  # divide for paralel computaion (propably)
    #         n_steps=51,  # default is 50
    #     ).squeeze(0)  # get rid of first dimention (remove if multiple proteins)
    #     deviations.append(attr)

    # dist_vectors = []
    # for dev in deviations:
    #     # compute per-row Euclidean distance
    #     dist = torch.sqrt(torch.sum((dev - attr_baseline) ** 2, dim=1))
    #     val = torch.sum(dist)
    #     dist_vectors.append(val)
    # print(dist_vectors)


def main(mdl: Classificator):
    example_inp = (
        "test_third",
        "KVKWFNNEKGYGFIEVEGEFIE",
    )

    captum_memory_probe(mdl, example_inp)


if __name__ == "__main__":
    TORCH_CUDA = "cuda"
    TORCH_CPU = "cpu"
    DEVICE = TORCH_CUDA if torch.cuda.is_available() else TORCH_CPU
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    N_STEPS = [50, 40, 30, 20, 10]

    main(classificator)
