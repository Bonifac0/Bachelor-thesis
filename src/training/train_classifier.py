import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
to run:
python -m src.training.train_classifier

because pyhon need to load packages
"""

# =========================
# Configuration
# =========================

X_PATH = "X.dat"  # memmap file
Y_PATH = "y.dat"  # memmap file
TOTAL_RESIDUES = 809  # change to your value
FEATURES = 1280

BATCH_SIZE = 128  # 2048
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 4

# =========================
# Load data (memmap)
# =========================

X = np.memmap(X_PATH, dtype=np.float16, mode="r", shape=(TOTAL_RESIDUES, FEATURES))

y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))

# =========================
# Dataset
# =========================


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


dataset = ResidueDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# =========================
# Model
# =========================


class ImportanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(FEATURES, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImportanceModel().to(device)

# =========================
# Loss (handle imbalance)
# =========================

pos = y.sum()
neg = y.shape[0] - pos
pos_weight = neg / max(pos, 1)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# Training loop
# =========================

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# =========================
# Save model
# =========================

torch.save(model.state_dict(), "importance_model.pt")
print("Model saved to importance_model.pt")
