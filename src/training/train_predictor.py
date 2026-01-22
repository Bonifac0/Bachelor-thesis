import numpy as np
import torch
import torch.nn as nn
from src.training.model_definitions import ImportancePredictor, DatasetHandler
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

EPOCHS = 50  # upper bound, early stopping will stop earlier
LR = 1e-2
WEIGHT_DECAY = 1e-5  # L2 regularization

DATASET_SPLIT = (0.8, 0.1, 0.1)

PATIENCE = 3
MIN_DELTA = 1e-3

# =========================
# Dataset
# =========================


X = np.memmap(X_PATH, dtype=np.float16, mode="r", shape=(TOTAL_RESIDUES, FEATURES))

y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))

dataset = DatasetHandler(X, y, DATASET_SPLIT)


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


print("a")
mean, std = mean_std(X)
print(f"Mean:{mean}")
print(f"Std:{std}")
print("b")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImportancePredictor(mean, std).to(device)

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

    for xb, yb in dataset.train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataset.train_loader)

    # -------- Validate --------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in dataset.val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

    val_loss /= len(dataset.val_loader)

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

# =========================
# Testing
# =========================

model.eval()
test_loss = 0.0

with torch.no_grad():
    for xb, yb in dataset.test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        test_loss += loss.item()
test_loss /= len(dataset.test_loader)

print(f"Testing loss: {test_loss}")
