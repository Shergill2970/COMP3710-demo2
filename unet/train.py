"""
Trains the UNet on OASIS brain MRI segmentation and reports per-class DSC.

Run from the project root:
    python3 unet/train.py
"""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_loader import OASISSegmentationDataset, TRAIN_IMG_DIR, TRAIN_SEG_DIR, \
    VAL_IMG_DIR, VAL_SEG_DIR, IMG_SIZE, NUM_SEG_CLASSES
from unet.model import UNet
from unet.metrics import dice_score_per_class, dice_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Hyperparameters ----
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
CE_WEIGHT = 0.5   # combined loss = CE_WEIGHT * CrossEntropy + (1 - CE_WEIGHT) * DiceLoss

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Data ----
train_set = OASISSegmentationDataset(TRAIN_IMG_DIR, TRAIN_SEG_DIR, img_size=IMG_SIZE, augment=False)
val_set = OASISSegmentationDataset(VAL_IMG_DIR, VAL_SEG_DIR, img_size=IMG_SIZE, augment=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print(f"Segmentation classes: {NUM_SEG_CLASSES}")

# ---- Model ----
model = UNet(in_channels=1, num_classes=NUM_SEG_CLASSES).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"UNet parameters: {n_params:,}")

ce_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# ---- Training loop ----
train_losses, val_losses = [], []
val_dice_history = []  # per-epoch list of per-class DSC, for the final report

start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)

        loss = CE_WEIGHT * ce_criterion(outputs, masks) + \
            (1 - CE_WEIGHT) * dice_loss(outputs, masks, NUM_SEG_CLASSES)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # ---- Validation: loss + per-class DSC ----
    model.eval()
    val_running = 0.0
    class_dice_sums = [0.0] * NUM_SEG_CLASSES
    n_val_batches = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)

            loss = CE_WEIGHT * ce_criterion(outputs, masks) + \
                (1 - CE_WEIGHT) * dice_loss(outputs, masks, NUM_SEG_CLASSES)
            val_running += loss.item() * images.size(0)

            batch_dice = dice_score_per_class(outputs, masks, NUM_SEG_CLASSES)
            for c in range(NUM_SEG_CLASSES):
                class_dice_sums[c] += batch_dice[c]
            n_val_batches += 1

    avg_val_loss = val_running / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    avg_class_dice = [s / n_val_batches for s in class_dice_sums]
    val_dice_history.append(avg_class_dice)

    scheduler.step(avg_val_loss)

    dice_str = " | ".join(f"class{c}: {d:.4f}" for c, d in enumerate(avg_class_dice))
    print(f"Epoch {epoch+1}/{EPOCHS} - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}")
    print(f"    Val DSC -> {dice_str} | mean: {sum(avg_class_dice)/len(avg_class_dice):.4f}")

duration = time.time() - start_time
print(f"\nTotal training time: {duration:.2f} seconds")

final_dice = val_dice_history[-1]
print("\nFinal per-class DSC on validation set:")
for c, d in enumerate(final_dice):
    status = "PASS" if d > 0.9 else "BELOW TARGET"
    print(f"  Class {c}: {d:.4f} ({status})")

# ---- Save checkpoint ----
checkpoint_path = os.path.join(OUTPUT_DIR, "unet_checkpoint.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "num_classes": NUM_SEG_CLASSES,
    "final_dice": final_dice,
}, checkpoint_path)
print("\nSaved checkpoint to", checkpoint_path)

# ---- Save loss curve ----
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Combined CE + Dice loss")
plt.title("UNet Training Loss")
plt.legend()
plt.grid(True)
loss_curve_path = os.path.join(OUTPUT_DIR, "unet_loss_curve.png")
plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
print("Saved loss curve to", loss_curve_path)

# ---- Save DSC-per-class-over-time curve ----
plt.figure(figsize=(8, 5))
val_dice_history_arr = list(zip(*val_dice_history))  # transpose: per-class series
for c, series in enumerate(val_dice_history_arr):
    plt.plot(series, label=f"Class {c}")
plt.axhline(0.9, color="red", linestyle="--", label="0.9 target")
plt.xlabel("Epoch")
plt.ylabel("DSC")
plt.title("Per-class DSC over training (validation set)")
plt.legend()
plt.grid(True)
dice_curve_path = os.path.join(OUTPUT_DIR, "unet_dice_curve.png")
plt.savefig(dice_curve_path, dpi=150, bbox_inches="tight")
print("Saved DSC curve to", dice_curve_path)
