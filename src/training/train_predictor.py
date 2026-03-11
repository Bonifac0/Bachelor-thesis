import numpy as np
import torch
import torch.nn as nn
from src.training.model_definitions import ImportancePredictor, DatasetHandler
import os
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.special import expit  # for sigmoid
from datetime import datetime

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
MIN_DELTA = 1e-6
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

wandb.init(
    project="importance-predictor",
    name=f"{MODE}_{timestamp}",
    config={
        "mode": MODE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "dataset_split": DATASET_SPLIT,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "model": "ImportancePredictor",
        "features": ImportancePredictor.FEATURES,
    },
)

config = wandb.config

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


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset and return loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())

    total_loss /= len(dataloader)
    all_logits = torch.cat(all_logits).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()

    probs = expit(all_logits)
    y_pred = (probs >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(all_labels, y_pred, zero_division=0),
        "recall": recall_score(all_labels, y_pred, zero_division=0),
        "f1": f1_score(all_labels, y_pred, zero_division=0),
        "accuracy": accuracy_score(all_labels, y_pred),
    }

    return total_loss, metrics


# -------- Training loop --------
for epoch in range(EPOCHS):
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

    # Validation
    val_loss, val_metrics = evaluate_model(model, dataset.val_loader, criterion, device)

    # Log everything to W&B
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"Accuracy: {val_metrics['accuracy']:.4f}"
    )

    # Early stopping
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# Save model
model_path = "resources/importance_model.pt"
torch.save(model.state_dict(), model_path)

artifact = wandb.Artifact(
    name="importance_model",
    type="model",
    metadata={
        "mode": MODE,
        "best_val_loss": best_val_loss,
    },
)
artifact.add_file(model_path)
wandb.log_artifact(artifact)
print("Model saved and logged to W&B")

# -------- Testing --------
test_loss, test_metrics = evaluate_model(model, dataset.test_loader, criterion, device)

wandb.log(
    {
        "test_loss": test_loss,
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
    }
)

print(
    f"Testing loss: {test_loss:.6f} | "
    f"Precision: {test_metrics['precision']:.4f} | "
    f"Recall: {test_metrics['recall']:.4f} | "
    f"F1: {test_metrics['f1']:.4f} | "
    f"Accuracy: {test_metrics['accuracy']:.4f}"
)
wandb.finish()
