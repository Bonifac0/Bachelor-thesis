import numpy as np
import torch
import torch.nn as nn
from src.training.model_definitions import ImportancePredictor, DatasetHandler
import os
from scipy.special import expit
from sklearn.metrics import precision_score

"""
to run:
python -m src.training.train_predictor
"""

# =========================
# Configuration
# =========================
# The model configuration in src/training/model_definiotion.py

MODE = "basic_1280"
X_PATH = f"training_data/{MODE}/X.dat"
Y_PATH = f"training_data/{MODE}/y.dat"

TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue

EPOCHS = 50  # upper bound
LR = 1e-2
WEIGHT_DECAY = 1e-5  # L2 regularization

DATASET_SPLIT = (0.6, 0.2, 0.2)

PATIENCE = 3
MIN_DELTA = 1e-5

# =========================
# Dataset
# =========================


X = np.memmap(
    X_PATH,
    dtype=np.float16,
    mode="r",
    shape=(TOTAL_RESIDUES, ImportancePredictor.FEATURES),
)
y = np.memmap(Y_PATH, dtype=np.uint8, mode="r", shape=(TOTAL_RESIDUES,))
dataset = DatasetHandler(X, y, DATASET_SPLIT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImportancePredictor().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataset.val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())

    val_loss /= len(dataset.val_loader)

    all_logits = torch.cat(all_logits).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()

    probs = expit(all_logits)
    y_pred = (probs >= 0.5).astype(int)

    val_precision = precision_score(all_labels, y_pred, zero_division=0)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Val Precision: {val_precision:.4f}")

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

all_logits = []
all_labels = []

with torch.no_grad():
    for xb, yb in dataset.test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        test_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(yb.cpu())

test_loss /= len(dataset.test_loader)

all_logits = torch.cat(all_logits).numpy().ravel()
all_labels = torch.cat(all_labels).numpy().ravel()

probs = expit(all_logits)
y_pred = (probs >= 0.5).astype(int)

test_precision = precision_score(all_labels, y_pred, zero_division=0)

print(f"Testing loss: {test_loss:.6f} | Testing precision: {test_precision:.4f}")
