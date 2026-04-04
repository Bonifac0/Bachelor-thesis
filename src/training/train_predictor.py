import numpy as np
import torch
import torch.nn as nn
from src.training.model_definitions import ImportancePredictor, DatasetHandler
import os
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.special import expit  # for sigmoid
from datetime import datetime
import json

"""
to run:
python -m src.training.train_predictor
"""


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

    return total_loss, metrics, y_pred, all_labels


def main():

    # =========================
    # Configuration
    # =========================
    # The model configuration in src/training/model_definitions.py

    # MODE = "basic_1280"
    MODE = "basic_1280_with_len"
    X_PATH = f"training_data/{MODE}/X.dat"
    Y_PATH = f"training_data/{MODE}/y.dat"
    LENGTHS_PATH = f"training_data/{MODE}/lengths.dat"
    AA_PATH = f"training_data/{MODE}/amino_acids.txt"

    TOTAL_RESIDUES = os.path.getsize(Y_PATH)  # uint8 -> 1 byte per residue

    EPOCHS = 50  # upper bound
    LR = 1e-3
    WEIGHT_DECAY = 1e-5  # L2 regularization

    DATASET_SPLIT = (0.6, 0.2, 0.2)

    PATIENCE = 3
    MIN_DELTA = 1e-6

    wandb.init(
        project="importance-predictor",
        name="basic_wlen_lr=e-3_hl=16",
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
    print("Starting training")

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
        val_loss, val_metrics, _, _ = evaluate_model(
            model, dataset.val_loader, criterion, device
        )

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

    # =========================
    # Save model + normalization and log to W&B
    # =========================

    model_path = "resources/importance_model.pt"

    # Save everything in one checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalization": dataset.norm_stats,  # mean/std for embeddings & length
        "features": ImportancePredictor.FEATURES,
        "mode": MODE,
    }

    # Save locally
    torch.save(checkpoint, model_path)

    # Create W&B artifact
    artifact = wandb.Artifact(
        name="importance_model",
        type="model",
        metadata={
            "mode": MODE,
            "best_val_loss": best_val_loss,
            "features": ImportancePredictor.FEATURES,
        },
    )

    # Add the local checkpoint file
    artifact.add_file(model_path)

    # Log artifact to W&B
    wandb.log_artifact(artifact)

    print(f"Model and normalization stats saved to {model_path} and logged to W&B")

    # -------- Testing --------
    test_loss, test_metrics, y_pred, y_true = evaluate_model(
        model, dataset.test_loader, criterion, device
    )

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

    # -------- Grouped Testing by Length --------
    print("\nTesting by Protein Length Groups:")

    residue_prot_lengths = np.memmap(
        LENGTHS_PATH,
        dtype=np.uint16,
        mode="r",
        shape=(TOTAL_RESIDUES,),
    )

    test_lengths = residue_prot_lengths[-len(y_pred) :]

    # Define groups (0-50, 51-100, ...)
    BIN_SIZE = 50
    max_len = int(np.max(test_lengths))
    bins = np.arange(0, max_len + BIN_SIZE, BIN_SIZE)

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (test_lengths > low) & (test_lengths <= high)

        if np.any(mask):
            y_pred_bin = y_pred[mask]
            y_true_bin = y_true[mask]

            acc = accuracy_score(y_true_bin, y_pred_bin)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

            group_name = f"test_group_{low + 1}-{high}"
            wandb.log(
                {
                    "group id": i,
                    "group name": group_name,
                    "group_test_accuracy": acc,
                    "group_test_f1": f1,
                    "group_test_precision": prec,
                    "group_test_recall": rec,
                    "group_test_count": np.sum(mask),
                }
            )

            print(
                f"Group {low + 1:3}-{high:3} | Samples: {np.sum(mask):6} | Acc: {acc:.4f} | F1: {f1:.4f}"
            )

    # -------- Grouped Testing by Amino Acid --------
    print("\nTesting by Amino Acid Groups:")

    # Load amino acid sequence (same order as residues)
    with open(AA_PATH, "r") as f:
        aa_sequence = f.read().strip()

    aa_array = np.array(list(aa_sequence))

    # Align with test set (same logic as lengths)
    test_aa = aa_array[-len(y_pred) :]

    # Standard amino acids
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    table = wandb.Table(
        columns=[
            "AA_name",
            "AA_f1",
            "AA_accuracy",
            "AA_pecision",
            "AA_recall",
            "AA_count",
            "run_name",
        ]
    )

    for i, aa in enumerate(AMINO_ACIDS):
        mask = test_aa == aa

        if np.any(mask):
            y_pred_bin = y_pred[mask]
            y_true_bin = y_true[mask]

            acc = accuracy_score(y_true_bin, y_pred_bin)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

            group_name = f"AA_{aa}"

            # wandb.log(
            #     {
            #         "AA_id": i,
            #         "AA_name": group_name,
            #         "AA_test_accuracy": acc,
            #         "AA_test_f1": f1,
            #         "AA_test_precision": prec,
            #         "AA_test_recall": rec,
            #         "AA_test_count": np.sum(mask),
            #     }
            # )
            table.add_data(
                group_name,
                f1,
                acc,
                prec,
                rec,
                np.sum(mask),
                wandb.run.name,  # important for grouping!
            )

            print(
                f"AA {aa} | Samples: {np.sum(mask):6} | Acc: {acc:.4f} | F1: {f1:.4f}"
            )
    wandb.log({"AA_table": table})

    wandb.finish()


if __name__ == "__main__":
    main()
