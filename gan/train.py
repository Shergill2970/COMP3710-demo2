"""
Trains a DCGAN on OASIS brain MRI slices (64x64) and saves training evidence:
loss curves and generated sample grids at regular intervals, as the
assignment requires for judging realism and checking for mode collapse.

Run from the project root:
    python3 gan/train.py
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
from data.data_loader import OASISImageDataset, TRAIN_IMG_DIR
from gan.model import Generator, Discriminator, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Hyperparameters ----
IMG_SIZE_GAN = 64      # smaller than VAE/UNet's 128 for GAN training stability
LATENT_DIM = 128
BATCH_SIZE = 128
EPOCHS = 100
LR = 2e-4               # standard DCGAN learning rate
BETA1 = 0.5              # standard DCGAN Adam beta1 (lower than default 0.9 - improves stability)
LABEL_SMOOTHING = 0.9    # real labels = 0.9 instead of 1.0 - a common mode-collapse mitigation
SAMPLE_EVERY = 5         # epochs between saved sample grids (training evidence)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ---- Data ----
train_set = OASISImageDataset(TRAIN_IMG_DIR, img_size=IMG_SIZE_GAN, augment=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
print(f"Train batches: {len(train_loader)}")

# ---- Models ----
generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

# Fixed noise vector so we can watch the SAME generated images evolve over
# training - the standard way to visually confirm progress and spot mode collapse
fixed_noise = torch.randn(64, LATENT_DIM, device=device)

g_losses, d_losses = [], []


def save_sample_grid(epoch):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).cpu()
    generator.train()

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for idx, ax in enumerate(axes.flat):
        img = (fake[idx, 0].numpy() + 1) / 2  # [-1,1] -> [0,1]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Generated samples - epoch {epoch}")
    path = os.path.join(SAMPLES_DIR, f"epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---- Training loop ----
start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    running_g_loss, running_d_loss = 0.0, 0.0

    for real_images in train_loader:
        real_images = real_images.to(device, non_blocking=True)
        batch_size = real_images.size(0)

        # =========================
        # Train Discriminator
        # =========================
        opt_d.zero_grad()

        real_labels = torch.full((batch_size,), LABEL_SMOOTHING, device=device)
        real_output = discriminator(real_images)
        d_loss_real = criterion(real_output, real_labels)

        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, device=device)
        fake_output = discriminator(fake_images.detach())  # detach: don't backprop into G here
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_d.step()

        # =========================
        # Train Generator
        # =========================
        opt_g.zero_grad()

        # Generator wants the discriminator to think fakes are real (label=1)
        gen_labels = torch.ones(batch_size, device=device)
        output = discriminator(fake_images)
        g_loss = criterion(output, gen_labels)
        g_loss.backward()
        opt_g.step()

        running_g_loss += g_loss.item() * batch_size
        running_d_loss += d_loss.item() * batch_size

    n = len(train_loader.dataset)
    avg_g_loss, avg_d_loss = running_g_loss / n, running_d_loss / n
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)

    print(f"Epoch {epoch}/{EPOCHS} - G loss: {avg_g_loss:.4f} - D loss: {avg_d_loss:.4f}")

    # Watch for mode collapse: D loss collapsing to ~0 while G loss explodes
    # is the classic warning sign - flagged here so it's visible during training,
    # not just discovered after the fact from the loss curve.
    if avg_d_loss < 0.05 and epoch > 10:
        print("    WARNING: discriminator loss very low - possible mode collapse risk. "
              "Consider reducing D's learning rate or training G more frequently.")

    if epoch % SAMPLE_EVERY == 0 or epoch == 1:
        save_sample_grid(epoch)

duration = time.time() - start_time
print(f"\nTotal training time: {duration:.2f} seconds")

# ---- Save checkpoints ----
torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, "discriminator.pth"))
print("Saved generator/discriminator checkpoints")

# ---- Save loss curves (required training evidence) ----
plt.figure(figsize=(8, 5))
plt.plot(g_losses, label="Generator loss")
plt.plot(d_losses, label="Discriminator loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Losses")
plt.legend()
plt.grid(True)
loss_curve_path = os.path.join(OUTPUT_DIR, "gan_loss_curve.png")
plt.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
print("Saved loss curve to", loss_curve_path)

# ---- Final high-quality sample grid ----
save_sample_grid(EPOCHS)
print(f"Sample grids saved throughout training in {SAMPLES_DIR}/ - "
      f"use these to show training progression and check for mode collapse.")
