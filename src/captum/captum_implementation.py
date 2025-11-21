from captum.attr import IntegratedGradients
from src.importance_vis import make_importance
from src.predictor import Classificator
import torch.nn.functional as F


"""
to run:
python -m src.captum.captum_implementation

because pyhon need to load Clasificator
"""


def captum(mdl: Classificator, inp):
    _, _, tokens = mdl.batch_converter(inp)
    embedding = mdl.model.embedding(tokens)

    ig = IntegratedGradients(mdl.model.forward_embedding)
    output = []
    for cls in range(4):
        attr, _ = ig.attribute(embedding, target=cls, return_convergence_delta=True)
        data = F.softmax(attr.sum(dim=2).squeeze(dim=0)[1:-1], dim=0).tolist()
        output.append(data)

    probability = mdl.classify(inp)[0]

    make_importance(inp[0][1], output, probability)


def main(mdl: Classificator):
    example_inp = [
        (
            "A0A512HC40",
            "SSRKVKWFNSEKSFSF",
        )
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
