"""
Shared data loading for the Preprocessed OASIS brain MRI dataset.

Used by all three tasks (VAE, UNet segmentation, GAN) so that data handling
is written once and stays consistent across the project.

CONFIRMED LAYOUT (verified directly against keras_png_slices_data.zip):

    keras_png_slices_data/
        keras_png_slices_train/          # 9664 grayscale image slices, PNG, 256x256
        keras_png_slices_test/           # 544 slices
        keras_png_slices_validate/       # 1120 slices
        keras_png_slices_seg_train/      # 9664 matching segmentation masks
        keras_png_slices_seg_test/       # 544 masks
        keras_png_slices_seg_validate/   # 1120 masks

Filenames: images are "case_XXX_slice_N.nii.png", masks are "seg_XXX_slice_N.nii.png".
sorted() on each directory's filename list produces matching image/mask pairs
(verified: 0 mismatches across all 9664 training pairs) because the "case_"/"seg_"
prefixes are constant-length within each folder and don't affect relative sort order.

Segmentation masks use FOUR classes, encoded as pixel values {0, 85, 170, 255}
(evenly spaced across the uint8 range, NOT plain class indices 0-3). This has been
verified across train/test/validate splits. SEG_LABEL_VALUES below maps these
raw pixel values to class indices 0-3 for one-hot encoding / cross-entropy loss.

On Rangpur, set ROOT_DIR to wherever the extracted keras_png_slices_data folder
lives, e.g. "/home/groups/comp3710/OASIS/keras_png_slices_data" - confirm the
exact path with `find /home/groups/comp3710/ -iname "*oasis*"` if unsure.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Paths - CONFIRM this path on Rangpur (folder structure inside it is verified)
# ---------------------------------------------------------------------------
ROOT_DIR = "/content/COMP3710-demo2/keras_png_slices_data"

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "keras_png_slices_train")
TEST_IMG_DIR = os.path.join(ROOT_DIR, "keras_png_slices_test")
VAL_IMG_DIR = os.path.join(ROOT_DIR, "keras_png_slices_validate")

TRAIN_SEG_DIR = os.path.join(ROOT_DIR, "keras_png_slices_seg_train")
TEST_SEG_DIR = os.path.join(ROOT_DIR, "keras_png_slices_seg_test")
VAL_SEG_DIR = os.path.join(ROOT_DIR, "keras_png_slices_seg_validate")

IMG_SIZE = 128  # source slices are 256x256; downsized for speed - raise if you have GPU budget

# Raw pixel values found in the segmentation masks, in class-index order.
# Verified via direct inspection: {0, 85, 170, 255}, 4 classes total.
SEG_LABEL_VALUES = [0, 85, 170, 255]
NUM_SEG_CLASSES = len(SEG_LABEL_VALUES)


# ---------------------------------------------------------------------------
# Dataset for VAE / GAN: images only, no labels needed
# ---------------------------------------------------------------------------
class OASISImageDataset(Dataset):
    """
    Loads plain MRI slice images (no segmentation labels).
    Used by Task 1 (VAE) and Task 3 (GAN).
    """

    def __init__(self, image_dir, img_size=IMG_SIZE, augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No PNG files found in {image_dir}. "
                f"Check ROOT_DIR / folder names at the top of data_loader.py "
                f"against the actual Rangpur directory structure."
            )

        tfs = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),  # MRI slices are single-channel
        ]
        if augment:
            # Keep augmentation light - these are anatomical scans, not natural
            # images, so avoid anything that distorts anatomy unrealistically.
            tfs.append(transforms.RandomHorizontalFlip())

        tfs += [
            transforms.ToTensor(),          # scales to [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # rescale to [-1, 1]
        ]
        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Dataset for UNet: image + segmentation mask pairs
# ---------------------------------------------------------------------------
class OASISSegmentationDataset(Dataset):
    """
    Loads MRI slice + matching segmentation mask pairs.
    Used by Task 2 (UNet).

    Returns:
        image: FloatTensor [1, H, W], normalized to [-1, 1]
        mask:  LongTensor  [H, W], integer class labels in {0, 1, 2, 3}
               - raw mask pixel values {0, 85, 170, 255} are remapped to
                 {0, 1, 2, 3} here, since cross-entropy / one-hot encoding
                 needs contiguous class indices, not raw pixel intensities.
               - converted to one-hot inside the training loop / loss function,
                 not here, so this Dataset stays reusable for both one-hot
                 and non-one-hot experiments.
    """

    def __init__(self, image_dir, seg_dir, img_size=IMG_SIZE, augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.seg_paths = sorted(glob.glob(os.path.join(seg_dir, "*.png")))

        if len(self.image_paths) == 0 or len(self.seg_paths) == 0:
            raise FileNotFoundError(
                f"No PNGs found in {image_dir} or {seg_dir}. "
                f"Check the ROOT_DIR / folder names at the top of data_loader.py."
            )
        if len(self.image_paths) != len(self.seg_paths):
            raise ValueError(
                f"Mismatched counts: {len(self.image_paths)} images vs "
                f"{len(self.seg_paths)} masks. Filenames may not be sorted "
                f"into matching pairs - verify naming convention on Rangpur."
            )

        self.img_size = img_size
        self.augment = augment

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        # Masks must use NEAREST resizing - bilinear would blend label values
        # into invalid in-between class indices.
        self.mask_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        mask = Image.open(self.seg_paths[idx])

        # NOTE: if you add random augmentation (flips/crops), the SAME random
        # transform must be applied to both img and mask together, or they'll
        # desync. Left out here for simplicity - add carefully if needed.

        img = self.img_transform(img)

        mask = self.mask_resize(mask)
        mask_arr = np.array(mask)

        # Remap raw pixel values {0, 85, 170, 255} -> class indices {0, 1, 2, 3}.
        class_mask = np.zeros_like(mask_arr, dtype=np.int64)
        for class_idx, pixel_val in enumerate(SEG_LABEL_VALUES):
            class_mask[mask_arr == pixel_val] = class_idx

        mask = torch.from_numpy(class_mask).long()

        return img, mask


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------
def get_vae_gan_loaders(batch_size=64, num_workers=4):
    train_set = OASISImageDataset(TRAIN_IMG_DIR, augment=True)
    val_set = OASISImageDataset(VAL_IMG_DIR, augment=False)
    test_set = OASISImageDataset(TEST_IMG_DIR, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_unet_loaders(batch_size=16, num_workers=4):
    train_set = OASISSegmentationDataset(TRAIN_IMG_DIR, TRAIN_SEG_DIR, augment=True)
    val_set = OASISSegmentationDataset(VAL_IMG_DIR, VAL_SEG_DIR, augment=False)
    test_set = OASISSegmentationDataset(TEST_IMG_DIR, TEST_SEG_DIR, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check - run this on Rangpur first to confirm paths/shapes
    # before writing any model code.
    print("Checking VAE/GAN image dataset...")
    train_loader, _, _ = get_vae_gan_loaders(batch_size=4)
    batch = next(iter(train_loader))
    print("Image batch shape:", batch.shape, "dtype:", batch.dtype,
          "min/max:", batch.min().item(), batch.max().item())

    print("\nChecking UNet segmentation dataset...")
    train_loader, _, _ = get_unet_loaders(batch_size=4)
    img_batch, mask_batch = next(iter(train_loader))
    print("Image batch shape:", img_batch.shape)
    print("Mask batch shape:", mask_batch.shape,
          "unique labels:", torch.unique(mask_batch))
