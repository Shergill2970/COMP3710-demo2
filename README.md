# COMP3710 Report: Generative and Segmentation Models on the OASIS Brain MRI Dataset

This repository implements three deep learning models applied to the Preprocessed OASIS
brain MRI dataset:

1. **Variational Autoencoder (VAE)** — learns a latent representation of brain MRI slices
   and visualises the resulting latent manifold.
2. **UNet** — segments brain MRI slices into anatomical structures, evaluated with the
   Dice Similarity Coefficient (DSC).
3. **Generative Adversarial Network (GAN)** — generates realistic synthetic brain MRI slices.

## Repository structure

```
comp3710-project/
├── data/
│   └── data_loader.py       # shared OASIS dataset loading for all three tasks
├── vae/                      # Task 1
├── unet/                     # Task 2
├── gan/                      # Task 3
├── utils/                    # shared helpers (plotting, checkpointing)
├── jobs/                     # SLURM job scripts for running on Rangpur
└── requirements.txt
```

## Dataset

The Preprocessed OASIS brain MRI dataset (`keras_png_slices_data`) is available on the
Rangpur cluster under `/home/groups/comp3710/OASIS/` (exact subpath to be confirmed via
`find /home/groups/comp3710/ -iname "*oasis*"` on first login). It is **not** included in
this repository (see `.gitignore`) due to size and access restrictions — all scripts read
directly from the cluster path.

**Structure** (verified directly against the dataset):

| Split | Images | Masks | Count |
|-------|--------|-------|-------|
| train | `keras_png_slices_train/` | `keras_png_slices_seg_train/` | 9,664 |
| test | `keras_png_slices_test/` | `keras_png_slices_seg_test/` | 544 |
| validate | `keras_png_slices_validate/` | `keras_png_slices_seg_validate/` | 1,120 |

- Images: grayscale (`L` mode) PNG, 256×256, filenames `case_XXX_slice_N.nii.png`.
- Masks: grayscale PNG, 256×256, filenames `seg_XXX_slice_N.nii.png`, matching each
  image 1:1 when both directories are sorted.
- Masks encode **4 segmentation classes** using raw pixel values `{0, 85, 170, 255}`
  (evenly spaced across the uint8 range) rather than plain class indices — the data
  loader remaps these to `{0, 1, 2, 3}` before one-hot encoding.

## Setup

**On Rangpur** (confirmed working setup for this project):

```bash
ssh <username>@rangpur.compute.eait.uq.edu.au
module load cuda/12.2
python3 -m venv ~/comp3710-env
source ~/comp3710-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

No `python`/`anaconda` module is needed — Rangpur's system Python (3.11.13) is used
directly, so scripts are run as `python3 script.py`, not `python script.py`.

Submit training jobs via SLURM using the `comp3710` partition (course-specific A100
nodes):

```bash
sbatch jobs/train_template.slurm
squeue -u $USER              # check job status
```

**Locally**, for development/testing without a cluster:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Task 1: Variational Autoencoder

<!-- TODO: architecture summary, training details (epochs, loss curve),
     and the latent manifold visualisation once complete -->

## Task 2: UNet Segmentation

<!-- TODO: architecture summary (skip connections, one-hot output), training details,
     per-label DSC scores table, and example segmentation visualisations -->

## Task 3: GAN

<!-- TODO: architecture summary, training curves (generator/discriminator loss),
     sample generated images across training, and notes on mode collapse if
     encountered -->

## Results summary

| Task | Metric | Result |
|------|--------|--------|
| VAE  | Manifold visualisation | <!-- link to image --> |
| UNet | Mean DSC across labels | <!-- e.g. 0.92 --> |
| GAN  | Qualitative realism | <!-- link to generated samples --> |

## Notes

- All models are implemented from scratch in PyTorch (no pretrained models used).
- Code is organised so that `data/data_loader.py` is shared across all three tasks
  to avoid duplicated data-handling logic.
