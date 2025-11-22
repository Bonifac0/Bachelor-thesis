import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


def lrp_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    relevance_out: torch.Tensor,
    eps: float = 1e-9,
    use_pos_only: bool = True,
) -> torch.Tensor:
    """
    LRP for a Linear layer with rule similar to Eq. (4)/(5).

    y = x @ weight.T
    x: (batch, ..., in_features)
    weight: (out_features, in_features)
    relevance_out: (batch, ..., out_features)
    """
    # Flatten all but last dim for easier math
    B = x.shape[0]
    x_flat = x.reshape(B, -1, x.shape[-1])  # (B, T, in_features)
    R_out_flat = relevance_out.reshape(
        B, -1, relevance_out.shape[-1]
    )  # (B, T, out_features)

    if use_pos_only:
        # Equivalent to restricting to q = {(i,j) | x_j w_ji >= 0}
        x_used = x_flat.clamp(min=0.0)
        w_used = weight.clamp(min=0.0)
    else:
        x_used = x_flat
        w_used = weight

    # z has shape (B, T, out_features, in_features)
    # z[b, t, i, j] ~ contribution of x_j to y_i
    z = x_used.unsqueeze(2) * w_used.unsqueeze(0).unsqueeze(0)

    # denominator sum_j' x_j' w_j'i (for each output i)
    denom = z.sum(dim=-1, keepdim=True) + eps  # (B, T, out_features, 1)

    # Messages from outputs i to inputs j
    # Multiply fraction of contribution by relevance_out
    message = z / denom * R_out_flat.unsqueeze(-1)  # (B, T, out_features, in_features)

    # Summation over outputs i gives relevance for each input j
    R_in_flat = message.sum(dim=2)  # (B, T, in_features)

    # Reshape back to original input shape
    R_in = R_in_flat.reshape_as(x)
    return R_in


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Will store forward-pass activations for LRP
        self.cache: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape into heads: (B, num_heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        scores = Qh @ Kh.transpose(-2, -1) / (self.head_dim**0.5)  # (B, H, T, T)
        attn = scores.softmax(dim=-1)  # attention weights
        head_out = attn @ Vh  # (B, H, T, head_dim)

        # Concat heads
        concat = head_out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.W_o(concat)  # (B, T, D)

        # Cache tensors needed for LRP
        self.cache["x"] = x
        self.cache["concat"] = concat
        self.cache["head_out"] = head_out
        self.cache["attn"] = attn

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int = 256, act: str = "gelu"
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)
        self.act = F.gelu if act == "gelu" else F.relu

        # Caches for LRP
        self.cache: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        attn_out = self.self_attn(x)
        x1 = self.norm1(x + attn_out)

        # Save for LRP
        self.cache["x_before_attn"] = x
        self.cache["x_after_attn"] = x1

        # Feedforward block
        ff1 = self.ffn1(x1)
        ff1_act = self.act(ff1)
        ff2 = self.ffn2(ff1_act)
        out = self.norm2(x1 + ff2)

        self.cache["ff1"] = ff1
        self.cache["ff1_act"] = ff1_act
        self.cache["ff2"] = ff2
        self.cache["x_after_ffn"] = out

        return out


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        max_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.embed(input_ids) + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x)

        # Simple pooling: mean over sequence
        pooled = x.mean(dim=1)  # (B, d_model)
        logits = self.classifier(pooled)
        return logits


class TransformerLRPExplainer:
    def __init__(self, model: TransformerClassifier):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def explain(
        self,
        input_ids: torch.Tensor,
        target_class: Optional[int] = None,
        use_pos_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "logits": logits,
                "token_relevance": R at last layer output (B, T, d_model),
                "head_scores": (num_layers, num_heads) aggregated relevance per head
            }
        """
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # 1. Forward pass
        logits = self.model(input_ids)  # (B, C)
        B, C = logits.shape

        if target_class is None:
            target_class = logits.argmax(dim=-1)  # (B,)
        else:
            target_class = torch.tensor(target_class, device=device).expand(B)

        # 2. Initialize relevance at output: R(0) = one-hot for each sample
        R_logits = torch.zeros_like(logits)
        R_logits[torch.arange(B), target_class] = 1.0

        # 3. LRP through classifier layer: pooled -> logits
        # pooled is x.mean(dim=1) in forward; we recompute for consistency
        with torch.no_grad():
            x = self.model.embed(input_ids)
            B, T, D = x.shape
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            x = x + self.model.pos_embed(pos)
            for layer in self.model.layers:
                x = layer(x)
            pooled = x.mean(dim=1)

        W_cls = self.model.classifier.weight  # (num_classes, d_model)
        R_pooled = lrp_linear(
            x=pooled,
            weight=W_cls,
            relevance_out=R_logits,
            use_pos_only=use_pos_only,
        )  # (B, d_model)

        # 4. From pooled back to last layer output (x_L): distribute relevance over time
        # Simple rule: average relevance across all time steps
        R_last_layer = R_pooled.unsqueeze(1).expand(B, T, D) / T  # (B, T, D)

        # 5. Backprop relevance through last Transformer block only (for demo)
        #    You can loop over all blocks in reverse similarly.
        last_block = self.model.layers[-1]

        # LRP through FFN: x_after_attn -> x_after_ffn
        ff1 = last_block.cache["ff1"]  # (B, T, d_ff)
        ff1_act = last_block.cache["ff1_act"]
        ff2 = last_block.cache["ff2"]

        W_ff2 = last_block.ffn2.weight  # (d_model, d_ff)
        R_ff1_act = lrp_linear(ff1_act, W_ff2, R_last_layer, use_pos_only=use_pos_only)

        # For GELU/ReLU, pass relevance through activation without change where activation positive
        # (simple z+ rule)
        mask = (ff1_act > 0).float()
        R_ff1 = R_ff1_act * mask

        W_ff1 = last_block.ffn1.weight  # (d_ff, d_model)
        x_after_attn = last_block.cache["x_after_attn"]
        R_after_attn = lrp_linear(
            x_after_attn, W_ff1, R_ff1, use_pos_only=use_pos_only
        )  # (B, T, d_model)

        # Residual in FFN block: out = norm2(x1 + ff2)
        # For simplicity, split relevance equally between x1 and ff2 inputs to norm2
        R_x1 = R_after_attn * 0.5
        R_ff2_residual = R_after_attn * 0.5

        # You could further refine this by inverting LayerNorm, etc.
        # For this minimal example, we focus on attention side.

        # 6. LRP through attention output projection: concat -> out
        attn_module = last_block.self_attn
        concat = attn_module.cache["concat"]  # (B, T, d_model)
        head_out = attn_module.cache["head_out"]  # (B, H, T, head_dim)

        W_o = attn_module.W_o.weight  # (d_model, d_model)
        R_concat = lrp_linear(
            x=concat, weight=W_o, relevance_out=R_x1, use_pos_only=use_pos_only
        )  # (B, T, d_model)

        # Map concat relevance back to heads
        H = attn_module.num_heads
        head_dim = attn_module.head_dim
        R_heads = R_concat.view(B, T, H, head_dim).transpose(
            1, 2
        )  # (B, H, T, head_dim)

        # Aggregate per-head relevance (sum over time and features)
        head_scores = R_heads.sum(dim=(2, 3))  # (B, H)
        # For multiple layers, you would repeat for each block in reversed order and stack

        result = {
            "logits": logits,
            "token_relevance": R_last_layer,  # relevance at last layer outputs
            "head_scores_last_layer": head_scores,  # per-head relevance for last block
        }

        return result


if __name__ == "__main__":
    # Example usage
    vocab_size = 30522
    model = TransformerClassifier(
        vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2, num_classes=5
    )

    explainer = TransformerLRPExplainer(model)

    # Fake batch of token ids
    input_ids = torch.randint(0, vocab_size, (2, 16))  # (batch=2, seq_len=16)

    out = explainer.explain(input_ids)

    print("Logits:", out["logits"].shape)  # (2, 5)
    print("Token relevance:", out["token_relevance"].shape)  # (2, 16, 128)
    print("Head scores (last layer):", out["head_scores_last_layer"].shape)  # (2, 4)
