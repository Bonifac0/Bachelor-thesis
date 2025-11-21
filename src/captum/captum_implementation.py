from captum.attr import IntegratedGradients
from src.importance_vis import make_importance
from src.predictor import Classificator
import torch.nn.functional as F


"""
to run:
python -m src.captum.captum_implementation

because pyhon need to load Clasificator
"""


def captum(mdl: Classificator, inp: list[tuple[str, str]]):
    _, _, tokens = mdl.batch_converter(inp)
    embedding = mdl.model.embedding(tokens)

    probability = mdl.classify(inp)
    ig = IntegratedGradients(mdl.model.forward_embedding)
    for i in range(len(inp)):  # iterate thru batch
        output = []
        for cls in range(4):  # for each class
            attr, _ = ig.attribute(
                embedding[i].unsqueeze(0), target=cls, return_convergence_delta=True
            )
            data = F.softmax(attr.sum(dim=2).squeeze(dim=0)[1:-1], dim=0).tolist()
            output.append(data)
            print("s")

        make_importance(inp[i], output, probability[i])


def main(mdl: Classificator):
    example_inp = [
        (
            "first",
            "SSRKVKWFNSEKSFSF",
        ),
        (
            "second",
            "EKGYGFIEVEGGRESF",
        ),
    ]
    cold_shock = [
        (
            "ori",
            "MQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKL",
        ),
        (
            "mut",
            "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA",
        ),
    ]
    PETase = [  # 32 mutaci
        (
            "wild-type",
            "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS",
        ),
        (
            "LK generated",
            "MNFPRASRLMQAAVLGGLMAVSAAATALTNPYARGPPPTAASLEASAGPFYVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIVIVLGYTARQSSIIWWGPRLASHGFVVITIITNSTLDQPSSRSSQALAALLQVLSLNGTSSSPIYYKVDNARMLVLGWSMGGGGSLILAANNESLKAAAPPAPWDSSTNFSSVTVPTLIIICENDSIAPVNSSALPIYYSMSRNAKQFLVIIGGSHSCANSSNSPQALIGKKYVAWWMRFMLNDTRYYTFACEPPNSTRVSDFYTANCS",
        ),
    ]

    captum(mdl, example_inp)


if __name__ == "__main__":
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    main(classificator)
