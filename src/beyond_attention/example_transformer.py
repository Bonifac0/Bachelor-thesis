import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]  # type: ignore


class AASequenceTransformerClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=64, n_heads=4, hidden_dim=128, num_classes=4
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.fc(pooled)


def encode_aa_sequence(seq, vocab):
    return torch.tensor([vocab.get(ch, vocab["X"]) for ch in seq], dtype=torch.long)


if __name__ == "__main__":
    aa_vocab = {
        "A": 0,
        "R": 1,
        "N": 2,
        "D": 3,
        "C": 4,
        "E": 5,
        "Q": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "L": 10,
        "K": 11,
        "M": 12,
        "F": 13,
        "P": 14,
        "S": 15,
        "T": 16,
        "W": 17,
        "Y": 18,
        "V": 19,
        "X": 20,
    }
    vocab_size = len(aa_vocab)

    model = AASequenceTransformerClassifier(vocab_size=vocab_size)

    # Example amino-acid sequences
    seq1 = "MKTLLILAV"
    seq2 = "ACDEFGHIKLMNPQRSTVWY"
    seq3 = "GGGGGGGGGG"

    # Encode and pad to same length
    encoded = [encode_aa_sequence(s, aa_vocab) for s in (seq1, seq2, seq3)]
    max_len = max(len(e) for e in encoded)
    padded = [
        torch.cat([e, torch.zeros(max_len - len(e), dtype=torch.long)]) for e in encoded
    ]

    batch = torch.stack(padded)  # (batch, seq_len)

    logits = model(batch)
    print(logits.shape)  # (3, 4)
    print(F.softmax(logits, dim=1))
