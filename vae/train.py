"""
Trains the VAE on OASIS brain MRI slices and saves a checkpoint + loss curve.

Run from the project root:
    python3 vae/train.py
"""

import os
import sys
import time
import matplotlib
matplotlib.use("Agg")  # no display available on a cluster
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_loader import OASISImageDataset, TRAIN_IMG_DIR, VAL_IMG_DIR, IMG_SIZE
from vae.model import VAE, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Hyperparameters ----
LATENT_DIM = 32        # use 2 if you want the classic direct-2D manifold grid instead of UMAP
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
KL_WEIGHT = 1.0         # reduce (e.g. 0.5) if reconstructions look blurry / KL dominates early

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Data ----
train_set = OASISImageDataset(TRAIN_IMG_DIR, img_size=IMG_SIZE, augment=False)
val_set = OASISImageDataset(VAL_IMG_DIR, img_size=IMG_SIZE, augment=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ---- Model ----
model = VAE(latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"VAE parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- Training loop ----
train_losses, val_losses = [], []

start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss, running_recon, running_kl = 0.0, 0.0, 0.0
    for images in train_loader:
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()
        recon, mu, logvar = model(images)
        loss, recon_loss, kl = vae_loss(recon, images, mu, logvar, kl_weight=KL_WEIGHT)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_recon += recon_loss.item() * images.size(0)
        running_kl += kl.item() * images.size(0)

    n = len(train_loader.dataset)
    avg_loss, avg_recon, avg_kl = running_loss / n, running_recon / n, running_kl / n
    train_losses.append(avg_loss)

    # ---- Validation ----
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device, non_blocking=True)
            recon, mu, logvar = model(images)
            loss, _, _ = vae_loss(recon, images, mu, logvar, kl_weight=KL_WEIGHT)
            val_running += loss.item() * images.size(0)
    avg_val_loss = val_running / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - train loss: {avg_loss:.2f} "
          f"(recon: {avg_recon:.2f}, kl: {avg_kl:.2f}) - val loss: {avg_val_loss:.2f}")

duration = time.time() - start_time
print(f"\nTotal training time: {duration:.2f} seconds")

# ---- Save checkpoint ----
checkpoint_path = os.path.join(OUTPUT_DIR, "vae_checkpoint.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "latent_dim": LATENT_DIM,
    "img_size": IMG_SIZE,
}, checkpoint_path)
print("Saved checkpoint to", checkpoint_path)

# ---- Save loss curve ----
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("ELBO loss (per sample)")
plt.title("VAE Training Loss")
plt.legend()
plt.grid(True)
loss_curve_path = os.path.join(OUTPUT_DIR, "vae_loss_curve.png")
plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
print("Saved loss curve to", loss_curve_path)

# ---- Save a quick reconstruction sample for a sanity check ----
model.eval()
with torch.no_grad():
    sample_batch = next(iter(val_loader))[:8].to(device)
    recon, _, _ = model(sample_batch)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    orig = (sample_batch[i, 0].cpu().numpy() + 1) / 2   # [-1,1] -> [0,1] for display
    rec = (recon[i, 0].cpu().numpy() + 1) / 2
    axes[0, i].imshow(orig, cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(rec, cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstruction")
plt.tight_layout()
recon_path = os.path.join(OUTPUT_DIR, "vae_reconstructions.png")
plt.savefig(recon_path, dpi=150, bbox_inches="tight")
print("Saved reconstruction sample to", recon_path)
