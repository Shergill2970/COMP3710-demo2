"""
Runs inference with a trained UNet on the OASIS test set and visualises the
results - this is the script to run live during the demonstration.

Run from the project root:
    python3 unet/predict.py [--index N]

With no arguments, it picks a handful of test images automatically and saves
a comparison figure (input / ground truth / prediction) plus prints DSC for
each shown example - use this to visually justify your DSC scores as the
assignment requires.
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_loader import OASISSegmentationDataset, TEST_IMG_DIR, TEST_SEG_DIR, IMG_SIZE
from unet.model import UNet
from unet.metrics import dice_score_per_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "unet_checkpoint.pth")


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    num_classes = checkpoint["num_classes"]
    model = UNet(in_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded UNet checkpoint - {num_classes} classes")
    print("Final training DSC per class:", checkpoint.get("final_dice"))
    return model, num_classes


def run_full_test_set_evaluation(model, num_classes):
    """Reports mean DSC per class across the WHOLE test set - the number to
    quote for your >0.9 DSC requirement, not just the few visualised examples."""
    test_set = OASISSegmentationDataset(TEST_IMG_DIR, TEST_SEG_DIR, img_size=IMG_SIZE, augment=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

    class_dice_sums = [0.0] * num_classes
    n_batches = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_dice = dice_score_per_class(outputs, masks, num_classes)
            for c in range(num_classes):
                class_dice_sums[c] += batch_dice[c]
            n_batches += 1

    mean_dice = [s / n_batches for s in class_dice_sums]
    print("\n=== Full test set DSC per class ===")
    for c, d in enumerate(mean_dice):
        status = "PASS" if d > 0.9 else "BELOW TARGET"
        print(f"  Class {c}: {d:.4f} ({status})")
    return mean_dice


def visualise_examples(model, num_classes, n_examples=4, indices=None):
    """Shows input / ground truth / prediction side by side for a handful of
    test images, with per-example DSC printed - direct visual justification
    of the DSC numbers, as the assignment requires."""
    test_set = OASISSegmentationDataset(TEST_IMG_DIR, TEST_SEG_DIR, img_size=IMG_SIZE, augment=False)

    if indices is None:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(test_set), size=n_examples, replace=False)

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        image, mask = test_set[idx]
        image_batch = image.unsqueeze(0).to(device)
        mask_batch = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_batch)
            pred = torch.argmax(output, dim=1)

        dice = dice_score_per_class(output, mask_batch, num_classes)

        img_display = (image[0].numpy() + 1) / 2  # [-1,1] -> [0,1]
        axes[row, 0].imshow(img_display, cmap="gray")
        axes[row, 0].set_title(f"Input (idx {idx})")
        axes[row, 1].imshow(mask.numpy(), cmap="viridis", vmin=0, vmax=num_classes - 1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(pred[0].cpu().numpy(), cmap="viridis", vmin=0, vmax=num_classes - 1)
        axes[row, 2].set_title(f"Prediction (mean DSC {sum(dice)/len(dice):.3f})")

        for col in range(3):
            axes[row, col].axis("off")

        print(f"Example idx {idx} - per-class DSC: " +
              " | ".join(f"class{c}: {d:.4f}" for c, d in enumerate(dice)))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "unet_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("\nSaved prediction visualisation to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None,
                         help="Specific test set index to run inference on (for the live demo)")
    args = parser.parse_args()

    model, num_classes = load_model()

    if args.index is not None:
        # Single-image demo mode: exactly what to run live during the demonstration
        visualise_examples(model, num_classes, n_examples=1, indices=[args.index])
    else:
        run_full_test_set_evaluation(model, num_classes)
        visualise_examples(model, num_classes, n_examples=4)
