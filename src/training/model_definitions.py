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

FEATURES = 1280
BATCH_SIZE = 2048
NUM_WORKERS = 4  # how it reads from disk


class ImportancePredictor(nn.Module):
    """
    The predictor itself
    """

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(FEATURES)
        self.linear = nn.Linear(FEATURES, 1)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x).squeeze(-1)


class ResidueDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class DatasetHandler:
    """
    Have train_loader, val_loader and test_loader
    Wraper for ResidueDataset and loaders
    """

    def __init__(self, X, y, dataset_split: tuple[float, float, float]):
        self.residue_dataset = ResidueDataset(X, y)
        self.num_samples = len(self.residue_dataset)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin = device.type == "cuda"
        train_set, val_set, test_set = self._prepare_loaders(dataset_split)

        self.train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

        self.test_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def _prepare_loaders(self, dataset_split: tuple[float, float, float]):
        assert sum(dataset_split) == 1
        indices = np.random.permutation(self.num_samples)

        train_end = int(dataset_split[0] * self.num_samples)
        val_end = train_end + int(dataset_split[1] * self.num_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_set = Subset(self.residue_dataset, train_idx)
        val_set = Subset(self.residue_dataset, val_idx)
        test_set = Subset(self.residue_dataset, test_idx)

        return train_set, val_set, test_set
