from captum.attr import IntegratedGradients
from src.importance_vis import make_importance
from src.predictor import Classificator
import torch.nn.functional as F
import torch


"""
to run:
python -m src.captum.captum_implementation

because pyhon need to load Clasificator
"""


def captum(mdl: Classificator, inp: list[tuple[str, str]]):
    probability = mdl.classify(inp)
    ig = IntegratedGradients(mdl.model.forward_embedding)

    for idx, input in enumerate(inp):  # iterate thru batch
        _, _, tokens = mdl.batch_converter([input])  # can not be done all at once
        embedding = mdl.model.embedding(tokens.to(DEVICE))
        output = []
        for cls in range(4):  # for each class
            attr, _ = ig.attribute(embedding, target=cls, return_convergence_delta=True)
            data = F.softmax(attr.sum(dim=2).squeeze(dim=0)[1:-1], dim=0).tolist()
            output.append(data)
            print("s")

        make_importance(inp[idx], output, probability[idx])


def main(mdl: Classificator):
    example_inp = [
        (
            "test_third",
            "SRPSGRGAGTVYYP",
        ),
        # (
        #     "first",
        #     "SSR",
        # ),
    ]
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
    cold_shock_long = [  # 34.5 GB gpu memory
        (  # 257,5MB/aminoacid
            "term",
            "MQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKLAMQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKLA",
        ),
        (
            "mezo",
            "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEAMQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKLA",
        ),
    ]
    PETase = [  # 32 mutaci # ~72.5GB gpu memory
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
    TORCH_CUDA = "cuda"
    TORCH_CPU = "cpu"
    DEVICE = TORCH_CUDA if torch.cuda.is_available() else TORCH_CPU
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    main(classificator)
