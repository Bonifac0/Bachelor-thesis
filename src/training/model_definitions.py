import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset


"""
improted in:
train_predictor.py
run_model.py
"""


class ImportancePredictorWithLengthAndHL(nn.Module):
    """
    Residue-level importance predictor for Captum embedding attributions + protein length.
    With hidel layer
    """

    FEATURES = 1281  # 1280 embedding + 1 length
    USE_LENGTH = True

    def __init__(self, hidden_dim: int = 16):
        super().__init__()

        # Small hidden layer to allow length-dependent adjustments
        self.model = nn.Sequential(
            nn.Linear(self.FEATURES, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1281) tensor, first 1280 = attribution, last = protein length
        returns: (N,) logits per residue
        """
        return self.model(x).squeeze(-1)


class ImportancePredictorWithLength(nn.Module):
    """
    Basic with protein length.
    """

    FEATURES = 1281  # 1280 embedding + 1 length
    USE_LENGTH = True

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(self.FEATURES, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class ImportancePredictorWith2HL(nn.Module):
    """
    Basic with hidel layer
    """

    FEATURES = 1280
    USE_LENGTH = False

    def __init__(self, first_hidden_dim: int = 64, second_hidden_dim: int = 16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.FEATURES, first_hidden_dim),
            nn.ReLU(),
            nn.Linear(first_hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class ImportancePredictorAllClassWithHL(nn.Module):
    """
    Basic with hidel layer
    """

    FEATURES = 1280 * 4
    USE_LENGTH = False

    def __init__(self, hidden_dim: int = 16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.FEATURES, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class ImportancePredictorWithHL(nn.Module):
    """
    Basic with hidel layer
    """

    FEATURES = 1280
    USE_LENGTH = False

    def __init__(self, hidden_dim: int = 16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.FEATURES, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


class ImportancePredictorWithNormalization(nn.Module):
    """
    Basic with normalization
    """

    FEATURES = 1280
    USE_LENGTH = False

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(self.FEATURES)
        self.linear = nn.Linear(self.FEATURES, 1)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x).squeeze(-1)


class ImportancePredictorBasic(nn.Module):
    """
    The basic one
    """

    FEATURES = 1280
    USE_LENGTH = False

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(self.FEATURES, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class ResidueDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        mean_atb,
        std_atb,
        mean_len=None,
        std_len=None,
    ):
        self.X = X
        self.y = y

        self.mean_atb = mean_atb
        self.std_atb = std_atb

        self.mean_len = mean_len
        self.std_len = std_len

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.mean_len is not None:
            atb = x[:1280]
            length = x[1280:]

            atb = (atb - self.mean_atb) / self.std_atb
            length = (length - self.mean_len) / self.std_len

            x = torch.cat([atb, length], dim=0)
        else:
            x = (x - self.mean_atb) / self.std_atb

        return x, y


class DatasetHandler:
    BATCH_SIZE = 2048
    NUM_WORKERS = 4

    def __init__(self, X, y, dataset_split, use_length=True):
        self.X = X
        self.y = y
        self.num_samples = len(X)
        self.use_length = use_length

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin = device.type == "cuda"

        train_idx, val_idx, test_idx = self._split_indices(dataset_split)

        # because of normalization
        X_train = X[train_idx].astype(np.float32)

        if self.use_length:
            X_atb = X_train[:, :1280]
            X_len = X_train[:, 1280:]

            mean_atb = torch.from_numpy(X_atb.mean(axis=0)).float()
            std_atb = torch.from_numpy(X_atb.std(axis=0) + 1e-8).float()

            mean_len = torch.from_numpy(X_len.mean(axis=0)).float()
            std_len = torch.from_numpy(X_len.std(axis=0) + 1e-8).float()

            self.norm_stats = {
                "mean_atb": mean_atb,
                "std_atb": std_atb,
                "mean_len": mean_len,
                "std_len": std_len,
            }
        else:
            mean_atb = torch.from_numpy(X_train.mean(axis=0)).float()
            std_atb = torch.from_numpy(X_train.std(axis=0) + 1e-8).float()

            self.norm_stats = {
                "mean_atb": mean_atb,
                "std_atb": std_atb,
            }

            mean_len = std_len = None  # not used

        print("Normalization finished")

        # =========================
        # Create datasets
        # =========================
        rd = ResidueDataset(X, y, mean_atb, std_atb, mean_len, std_len)
        train_set = Subset(
            rd,
            train_idx,
        )
        val_set = Subset(
            rd,
            val_idx,
        )
        test_set = Subset(
            rd,
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

        return (
            indices[:train_end],
            indices[train_end:val_end],
            indices[val_end:],
        )
