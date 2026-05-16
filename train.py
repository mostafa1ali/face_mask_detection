"""
train.py — Training Pipeline
==============================
Usage:
    python train.py

What happens:
    1. Load data from data/train/ and data/val/
    2. Build the CNN model
    3. Train for N epochs with Adam optimizer + CrossEntropyLoss
    4. Save the best model to mask_cnn.pth
    5. Plot training / validation curves
"""

import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model   import MaskCNN
from dataset import get_dataloaders


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LEARNING_RATE = 1e-3
SAVE_PATH   = "mask_cnn.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    """Run one full pass over the training set."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # 1. clear old gradients
        outputs = model(images)         # 2. forward pass  → logits (B, 2)
        loss    = criterion(outputs, labels)  # 3. compute loss
        loss.backward()                 # 4. backprop
        optimizer.step()                # 5. update weights

        total_loss += loss.item() * images.size(0)
        _, preds    = torch.max(outputs, 1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate on validation set (no gradient updates)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds    = torch.max(outputs, 1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses,   label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(val_accs,   label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    print("Training curves saved to training_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}\n")

    # 1. Data
    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

    # 2. Model
    model = MaskCNN().to(DEVICE)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler — reduces LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=3, factor=0.5)

    # 4. Training Loop
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)

        scheduler.step(vl_loss)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, SAVE_PATH)
            print(f"  ✓ New best model saved  (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to '{SAVE_PATH}'")

    # 5. Plot curves
    plot_curves(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()
