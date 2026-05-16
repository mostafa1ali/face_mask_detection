"""
model.py — CNN Architecture for Face Mask Detection
====================================================
Similar concept to CNNonFMNIST but adapted for binary classification:
    Class 0 → With Mask
    Class 1 → Without Mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskCNN(nn.Module):
    """
    Convolutional Neural Network for face mask detection.

    Architecture:
        Input  : (B, 3, 64, 64)   ← RGB face crop, resized to 64×64
        Conv1  : 32 filters, 3×3  → (B, 32, 62, 62) → MaxPool → (B, 32, 31, 31)
        Conv2  : 64 filters, 3×3  → (B, 64, 29, 29) → MaxPool → (B, 64, 14, 14)
        Conv3  : 128 filters, 3×3 → (B, 128, 12, 12) → MaxPool → (B, 128, 6, 6)
        Flatten: 128 × 6 × 6 = 4608
        FC1    : 512 neurons  + ReLU + Dropout(0.5)
        FC2    : 2   neurons  (logits for 2 classes)
    """

    def __init__(self):
        super(MaskCNN, self).__init__()

        # ── Convolutional Blocks ──────────────────────────────────────────
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),   # (B,32,62,62)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                        # (B,32,31,31)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # (B,64,29,29)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                        # (B,64,14,14)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), # (B,128,12,12)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                        # (B,128,6,6)
        )

        # ── Fully Connected Layers ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),          # Dropout to reduce overfitting
            nn.Linear(512, 2)           # 2 output classes
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x  # raw logits — use CrossEntropyLoss (includes softmax)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MaskCNN()
    print(model)

    dummy = torch.randn(8, 3, 64, 64)   # batch of 8 RGB 64×64 images
    out   = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # should be (8, 2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
