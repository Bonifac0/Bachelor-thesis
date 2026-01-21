import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os

"""
to run:
python -m src.training.train_predictor
"""

# =========================
# Configuration
# =========================

X_PATH = "X.dat"
Y_PATH = "y.dat"

TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue
FEATURES = 1280

BATCH_SIZE = 2048
EPOCHS = 50  # upper bound, early stopping will stop earlier
LR = 1e-2
WEIGHT_DECAY = 1e-5  # L2 regularization
NUM_WORKERS = 4

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

PATIENCE = 3
MIN_DELTA = 1e-3

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


X = np.memmap(X_PATH, dtype=np.float16, mode="r", shape=(TOTAL_RESIDUES, FEATURES))

y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))

dataset = ResidueDataset(X, y)

# =========================
# Train / Val / Test split
# =========================

num_samples = len(dataset)
indices = np.random.permutation(num_samples)

train_end = int(TRAIN_FRAC * num_samples)
val_end = train_end + int(VAL_FRAC * num_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
# =========================
# Normalizer
# =========================


def mean_std(set):
    CHUNK_SIZE = 10_000

    mean = np.zeros(FEATURES, dtype=np.float64)
    M2 = np.zeros(FEATURES, dtype=np.float64)
    count = 0

    for start in range(0, TOTAL_RESIDUES, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, TOTAL_RESIDUES)
        chunk = set[start:end].astype(np.float64)

        chunk_count = chunk.shape[0]
        chunk_mean = chunk.mean(axis=0)
        chunk_var = chunk.var(axis=0)

        delta = chunk_mean - mean
        new_count = count + chunk_count

        mean += delta * chunk_count / new_count
        M2 += chunk_var * chunk_count + (delta**2) * count * chunk_count / new_count

        count = new_count

        if start % (CHUNK_SIZE * 10) == 0:
            print(f"Processed {count} residues")

    std = np.sqrt(M2 / count) + 1e-6

    return mean.astype(np.float32), std.astype(np.float32)


# =========================
# Model
# =========================


class ImportanceModel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        self.norm = nn.LayerNorm(FEATURES)
        self.linear = nn.Linear(FEATURES, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.norm(x)
        return self.linear(x).squeeze(-1)


print("a")
mean, std = mean_std(X)
print(mean)
print(std)
print("b")


# mean = X.mean(axis=0)
# std = X.std(axis=0) + 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImportanceModel(mean, std).to(device)

# =========================
# Loss (class imbalance)
# =========================

pos = y.sum()
neg = y.shape[0] - pos
pos_weight = neg / max(pos, 1)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

# =========================
# Optimizer (L2 regularization)
# =========================

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# =========================
# Training with Early Stopping
# =========================

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    # -------- Train --------
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- Validate --------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # -------- Early stopping --------
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

torch.save(model.state_dict(), "importance_model.pt")
print("Model saved")


# Testing
model.eval()
test_loss = 0.0

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        test_loss += loss.item()
test_loss /= len(test_loader)

print(f"Testing loss: {test_loss}")
