import esm
import torch
import typing
from collections import OrderedDict
import torch.nn.functional as F


MODEL_8M = "esm2_t6_8M_UR50D"
MODEL_35M = "esm2_t12_35M_UR50D"
MODEL_150M = "esm2_t30_150M_UR50D"
MODEL_650M = "esm2_t33_650M_UR50D"
MODEL_3B = "esm2_t36_3B_UR50D"
MODEL_15B = "esm2_t48_15B_UR50D"
TORCH_CUDA = "cuda"
TORCH_CPU = "cpu"
DEVICE = TORCH_CUDA if torch.cuda.is_available() else TORCH_CPU


class Classificator:
    def __init__(self, model_path: str) -> None:
        self.MODEL_PATH = model_path
        self.model, self.batch_converter = self.prepare_model()

    def classify(self, inp: typing.List[typing.Tuple[str, str]]) -> typing.List:
        """
        Return matrix 4*BatchSize of probabilities.
        """
        _, _, inputs = self.batch_converter(inp)
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            output = self.model(inputs)[0]
        return F.softmax(output.detach().to("cpu"), dim=1).tolist()

    def load_pretrained_model(self, model_name: str, torch_device: str):
        # model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        if model_name == MODEL_8M:
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif model_name == MODEL_35M:
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif model_name == MODEL_150M:
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif model_name == MODEL_650M:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model_name == MODEL_3B:
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        elif model_name == MODEL_15B:
            model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        map = alphabet.tok_to_idx
        rev_map = {map[key]: key for key in map}
        batch_converter = alphabet.get_batch_converter()
        device = torch.device(torch_device)
        model = model.to(device)
        return model, map, rev_map, batch_converter, device, alphabet

    def prepare_model(self):
        print("loading pretrained model (scaffolding)...")
        model_pretrain, _, _, batch_converter, _, alphabet = self.load_pretrained_model(
            model_name=MODEL_650M, torch_device=TORCH_CPU
        )
        print("building model...")
        model = ModelClassifier(
            embedding=model_pretrain.embed_tokens,
            layers=model_pretrain.layers,
            emb_layer_norm_after=model_pretrain.emb_layer_norm_after,
            alphabet=alphabet,
            dropout=0,
        ).to(DEVICE)

        print("loading finetuned weights...")
        checkpoint = torch.load(self.MODEL_PATH, map_location=TORCH_CPU)
        checkpoint_2 = OrderedDict(
            (k.replace("module.", ""), v)
            for k, v in checkpoint["model_state_dict"].items()
        )
        model.load_state_dict(checkpoint_2)
        model.to(DEVICE)
        model.eval()
        print("model loaded")
        return model, batch_converter


class ModelClassifier(torch.nn.Module):
    def __init__(
        self,
        embedding,
        layers,
        emb_layer_norm_after,
        alphabet: typing.Union[esm.data.Alphabet, str] = "ESM-1b",
        dropout: float = 0.0,
        classes: int = 4,
        noise_std: float = 0.0,
        noise_adaptive: bool = True,
        fill_mask: bool = False,
        freeze: int = 0,
    ):
        super().__init__()
        self.embedding = embedding
        self.layers = layers
        self.emb_layer_norm_after = emb_layer_norm_after
        self.out_categorical = torch.nn.Linear(embedding.embedding_dim, classes)
        self.out_binary = torch.nn.Linear(embedding.embedding_dim, 1)
        self.out_regression = torch.nn.Linear(embedding.embedding_dim, 1)

        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.continue_idx = alphabet.tok_to_idx["-"]
        self.embed_scale = 1
        self.noise_std = noise_std
        self.noise_adaptive = noise_adaptive
        self.fill_mask = fill_mask

        for i, layer in enumerate(self.layers):
            if i >= freeze:
                layer.self_attn.dropout = dropout
                layer.requires_grad_(True)
            else:
                layer.requires_grad_(False)

        if noise_std > 0 or freeze > 0:
            self.embedding.requires_grad_(False)

    def forward(self, tokens, input_mask=None):
        assert tokens.ndim == 2
        if input_mask is not None:
            assert input_mask.ndim == 3
        padding_mask = tokens.eq(self.padding_idx)  # B, T
        cls_mask = tokens.eq(self.cls_idx)
        eos_mask = tokens.eq(self.eos_idx)
        continue_mask = tokens.eq(self.continue_idx)
        all_mask = torch.logical_or(
            padding_mask,
            torch.logical_or(cls_mask, torch.logical_or(eos_mask, continue_mask)),
        )
        all_mask = all_mask.unsqueeze(dim=2)

        # input to embedding
        x = self.embed_scale * self.embedding(tokens)

        # apply mask to embedding & add noise
        if input_mask is not None:
            device = next(self.parameters()).device
            x = x * input_mask

            if self.noise_adaptive:
                noise_mask = 1 - torch.pow(input_mask, 10)
                noise = torch.randn(x.shape).to(device) * self.noise_std
                noise = (noise * noise_mask).to(device)
            else:
                noise = torch.randn(x.shape).to(device) * self.noise_std
            x = x + noise

            if self.fill_mask:
                mask_emb = (
                    torch.ones(x.shape[:2], dtype=torch.int32).to(device)
                    * self.mask_idx
                )
                mask_emb = self.embedding(mask_emb)
                reversed_mask = 1 - input_mask
                mask_emb = mask_emb * reversed_mask
                if self.noise_adaptive:
                    noise_mask = 1 - torch.pow(input_mask, 10)
                    noise = torch.randn(x.shape).to(device) * self.noise_std
                    noise = (noise * noise_mask).to(device)
                else:
                    noise = torch.randn(x.shape).to(device) * self.noise_std
                mask_emb = mask_emb + noise
                x = x + mask_emb

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        i = 0
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
            )
            i += 1
        x = self.emb_layer_norm_after(x)

        out_cat = self.out_categorical(x[0, :, :])
        out_bin = self.out_binary(x[0, :, :])
        out_reg = self.out_regression(x[0, :, :])

        return [out_cat, out_bin, out_reg]


if __name__ == "__main__":
    import torch.nn.functional as F

    MODEL_PATH = "resources/model-664.pt"  # .pt file
    classificator = Classificator(MODEL_PATH)

    example_inp = [
        (
            "A0A512HC40",
            "MSVGVGCSSSCSPETLAALVRATLAEAAVPLDRIACIATLDRRVPHPAVQGLARALGGVPVRGFSPETLNAVAPERLRTVSEKTRQTVGCASVAEAAALCALGSRARLLIPRRADARATCAVATSPSHGP",
        )
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
    outputs = classificator.classify(PETase)
    print(outputs)
