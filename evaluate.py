"""
evaluate.py — Model Evaluation & Confusion Matrix
===================================================
After training, use this script to:
    • Compute accuracy, precision, recall, F1 on validation set
    • Plot confusion matrix
    • Visualize sample predictions

Usage:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

from model   import MaskCNN
from dataset import get_dataloaders, CLASS_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "mask_cnn.pth"
DATA_DIR   = "data"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = MaskCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def get_predictions(model, loader):
    """Run the full validation set through the model and collect results."""
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)                         # (B, 2)
            probs  = torch.softmax(logits, dim=1)          # (B, 2)
            _, preds = torch.max(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]   # ["With Mask", "Without Mask"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Face Mask Detection")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Saved confusion_matrix.png")

    # Compute per-cell stats for report
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (Mask correctly → Mask)      : {tp}")
    print(f"  True Negatives  (No Mask correctly → No Mask): {tn}")
    print(f"  False Positives (No Mask → predicted Mask)   : {fp}")
    print(f"  False Negatives (Mask → predicted No Mask)   : {fn}")


def main():
    print(f"Loading model from '{MODEL_PATH}'...")
    model = load_model()

    _, val_loader = get_dataloaders(DATA_DIR, batch_size=32)

    print("Running inference on validation set...")
    y_true, y_pred, y_probs = get_predictions(model, val_loader)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]
    ))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()
