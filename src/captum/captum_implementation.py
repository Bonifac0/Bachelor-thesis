from captum.attr import IntegratedGradients
from src.importance_vis import make_importance
from src.predictor import Classificator
import torch.nn.functional as F


"""
to run:
python -m src.captum.captum_implementation

because pyhon need to load Clasificator
"""


def main(mdl: Classificator):
    example_inp = [
        (
            "A0A512HC40",
            "SSRKVKWFNSEKGFGF",
        )
    ]
    cold_shock = [
        ("ori", "MQRGKVKWFNNEKGYGFIEVEGGSDVFVHFTAIQGEGFKTLEEGQEVSFEIVQGNRGPQAANVVKL"),
        ("mut", "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA"),
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
    _, _, tokens = mdl.batch_converter(example_inp)
    # print(tokens)
    embedding = mdl.model.embedding(tokens)
    # output = mdl.model.forward_embedding(embedding)
    # print(output)
    # sftmax = F.softmax(output.detach().to("cpu"), dim=1).tolist()
    # print(sftmax)

    ig = IntegratedGradients(mdl.model.forward_embedding)
    attr, _ = ig.attribute(embedding, target=0, return_convergence_delta=True)
    print(attr.sum(dim=2))
    data = F.softmax(attr.sum(dim=2).squeeze(dim=0)[1:-1]).tolist()
    print(data)
    make_importance(example_inp[0][1], data)

    # attr, _ = ig.attribute(embedding, target=1, return_convergence_delta=True)
    # print(attr.sum(dim=2))
    # outputs = classificator.classify(example_inp)
    # print(outputs)


if __name__ == "__main__":
    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    main(classificator)
