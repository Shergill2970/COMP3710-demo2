"""
Visualises the VAE's latent manifold.

- If latent_dim == 2: directly samples a 2D grid of z values across a range
  of the standard normal prior, decodes each into an image, and tiles them
  into a single manifold image (the classic VAE visualisation).
- If latent_dim > 2: encodes the test set, reduces the latent vectors to 2D
  with UMAP, and produces a scatter plot of the latent space, plus a small
  grid of random prior samples decoded into images as a qualitative check.

Run from the project root (after vae/train.py has produced a checkpoint):
    python3 vae/visualise_manifold.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_loader import OASISImageDataset, TEST_IMG_DIR, IMG_SIZE
from vae.model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "vae_checkpoint.pth")

# ---- Load trained model ----
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
latent_dim = checkpoint["latent_dim"]
img_size = checkpoint["img_size"]

model = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded VAE checkpoint - latent_dim={latent_dim}")


def plot_2d_grid_manifold(n=20, span=3.0):
    """
    Classic VAE manifold visualisation for latent_dim == 2: sample a grid of
    (z1, z2) values across [-span, span] (covering most of the standard
    normal's probability mass), decode each into an image, and tile them.
    """
    grid_x = np.linspace(-span, span, n)
    grid_y = np.linspace(-span, span, n)[::-1]  # flip so origin is bottom-left visually

    canvas = np.zeros((n * img_size, n * img_size))
    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                z = torch.tensor([[x, y]], dtype=torch.float32).to(device)
                decoded = model.decoder(z)
                img = (decoded[0, 0].cpu().numpy() + 1) / 2  # [-1,1] -> [0,1]
                canvas[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas, cmap="gray")
    plt.title(f"VAE Latent Manifold (2D grid, span=±{span})")
    plt.axis("off")
    out_path = os.path.join(OUTPUT_DIR, "vae_manifold_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved 2D grid manifold to", out_path)


def plot_umap_manifold():
    """
    For latent_dim > 2: encode the test set, reduce to 2D with UMAP, and
    scatter-plot the resulting embedding. Also decodes a grid of random prior
    samples as a qualitative "does this look like a manifold of brains" check.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for latent_dim > 2 visualisation. "
            "Install with: pip install umap-learn"
        )

    test_set = OASISImageDataset(TEST_IMG_DIR, img_size=img_size, augment=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    all_mu = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            mu, _ = model.encoder(images)
            all_mu.append(mu.cpu().numpy())
    all_mu = np.concatenate(all_mu, axis=0)
    print(f"Encoded {all_mu.shape[0]} test images into {latent_dim}-D latent space")

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_mu)

    plt.figure(figsize=(8, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.5)
    plt.title(f"UMAP projection of VAE latent space (from {latent_dim}D)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    scatter_path = os.path.join(OUTPUT_DIR, "vae_umap_latent.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print("Saved UMAP latent scatter to", scatter_path)

    # Qualitative check: decode a grid of random prior samples
    n = 8
    with torch.no_grad():
        z = torch.randn(n * n, latent_dim).to(device)
        decoded = model.decoder(z)

    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        img = (decoded[idx, 0].cpu().numpy() + 1) / 2
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle("Random samples decoded from the prior N(0, I)")
    samples_path = os.path.join(OUTPUT_DIR, "vae_prior_samples.png")
    plt.savefig(samples_path, dpi=150, bbox_inches="tight")
    print("Saved prior sample grid to", samples_path)


if __name__ == "__main__":
    if latent_dim == 2:
        plot_2d_grid_manifold()
    else:
        plot_umap_manifold()
