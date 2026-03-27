import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset


"""
improted in:
train_predictor.py
run_model.py
predictor_tester.py
"""


class ImportancePredictor(nn.Module):
    """
    Residue-level importance predictor for Captum embeddings + protein length.
    """

    FEATURES = 1281  # 1280 embeddings + 1 length

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Small hidden layer to allow length-dependent adjustments
        self.model = nn.Sequential(
            nn.Linear(self.FEATURES, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1281) tensor, first 1280 = embeddings, last = protein length
        returns: (N,) logits per residue
        """
        return self.model(x).squeeze(-1)


class ResidueDataset(Dataset):
    def __init__(self, X, y, mean_emb, std_emb, mean_len, std_len):
        self.X = X
        self.y = y

        self.mean_emb = mean_emb
        self.std_emb = std_emb
        self.mean_len = mean_len
        self.std_len = std_len

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        # Separate handeling of different features type
        emb = x[:1280]
        length = x[1280:]

        # normalize
        emb = (emb - self.mean_emb) / self.std_emb
        length = (length - self.mean_len) / self.std_len

        x = torch.cat([emb, length], dim=0)

        return x, y


class DatasetHandler:
    """
    Have train_loader, val_loader and test_loader
    Wraper for ResidueDataset and loaders
    """

    BATCH_SIZE = 2048
    NUM_WORKERS = 4  # how it reads from disk

    def __init__(self, X, y, dataset_split):
        self.X = X
        self.y = y
        self.num_samples = len(X)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin = device.type == "cuda"

        train_idx, val_idx, test_idx = self._split_indices(dataset_split)

        # =========================
        # Compute normalization (TRAIN ONLY)
        # =========================
        X_train = X[train_idx].astype(np.float32)  # convert to float32 temporarily

        X_emb = X_train[:, :1280]
        X_len = X_train[:, 1280:]

        mean_emb = torch.from_numpy(X_emb.mean(axis=0)).float()
        std_emb = torch.from_numpy(X_emb.std(axis=0) + 1e-8).float()

        mean_len = torch.from_numpy(X_len.mean(axis=0)).float()
        std_len = torch.from_numpy(X_len.std(axis=0) + 1e-8).float()

        # store for later (saving model)
        self.norm_stats = {
            "mean_emb": mean_emb,
            "std_emb": std_emb,
            "mean_len": mean_len,
            "std_len": std_len,
        }
        print("Normalization finnished")

        # =========================
        # Create datasets
        # =========================
        train_set = Subset(
            ResidueDataset(X, y, mean_emb, std_emb, mean_len, std_len),
            train_idx,
        )
        val_set = Subset(
            ResidueDataset(X, y, mean_emb, std_emb, mean_len, std_len),
            val_idx,
        )
        test_set = Subset(
            ResidueDataset(X, y, mean_emb, std_emb, mean_len, std_len),
            test_idx,
        )

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.NUM_WORKERS,
            pin_memory=pin,
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=pin,
        )

        self.test_loader = DataLoader(
            test_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            pin_memory=pin,
        )

    def _split_indices(self, dataset_split):
        assert np.isclose(sum(dataset_split), 1.0)

        indices = np.arange(self.num_samples)

        train_end = int(dataset_split[0] * self.num_samples)
        val_end = train_end + int(dataset_split[1] * self.num_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return train_idx, val_idx, test_idx
