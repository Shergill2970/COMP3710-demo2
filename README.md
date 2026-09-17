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

**Architecture**: convolutional encoder (4 strided conv layers, 128→64→32→16→8 spatial
downsampling) feeding two linear heads (`fc_mu`, `fc_logvar`), a reparameterisation step
(`z = mu + std * epsilon`), and a mirrored transposed-conv decoder (8→16→32→64→128)
ending in `tanh` to match the `[-1, 1]` normalised input images. Latent dimension: 32.

**Training**: 30 epochs, combined reconstruction (MSE) + KL-divergence (ELBO) loss,
Adam optimiser (lr 1e-3). Training took **463.5 seconds** (~7.7 minutes) on an A100.
The KL term stabilised around ~42 rather than collapsing to 0, indicating the latent
code was being used meaningfully rather than ignored (no posterior collapse).

**Results**:
- Reconstructions closely resemble real brain MRI slices, including ventricle structure
  and skull outline, with the mild blurriness expected of a plain VAE — see
  `vae/outputs/vae_reconstructions.png`.
- Latent manifold visualised via UMAP (latent_dim=32 > 2), producing distinct curved
  clusters — consistent with each OASIS case's stack of adjacent slices tracing a
  smooth path through latent space — see `vae/outputs/vae_umap_latent.png`.
- Samples decoded from pure `N(0, I)` noise (no real image encoded first) produce
  recognisable, diverse brain anatomy, confirming the model learned a genuine
  generative manifold rather than just memorising reconstructions — see
  `vae/outputs/vae_prior_samples.png`.

## Task 2: UNet Segmentation

**Architecture**: standard UNet — 4-stage contracting path (`DoubleConv` + `MaxPool`
blocks, channels 64→128→256→512), a bottleneck (1024 channels), and a matching
4-stage expanding path using transposed convolutions with **skip connections**
concatenating each encoder stage's feature map into the corresponding decoder stage.
Output: 4 channels (one per class), i.e. **categorical/one-hot-style output** — softmax
+ argmax applied at evaluation time, not inside the model, so training uses
`CrossEntropyLoss` directly on raw logits.

**Training**: 40 epochs, combined loss (50% `CrossEntropyLoss` + 50% differentiable
Dice loss), Adam optimiser (lr 1e-3) with `ReduceLROnPlateau` scheduling. Training took
**5324.0 seconds** (~88.7 minutes) on an A100. Validation loss/DSC plateaued around
epoch 17–22 before mild overfitting set in (train loss kept falling while val loss
crept back up slightly) — the LR scheduler stabilised performance rather than letting
it degrade further.

**Results — final per-class DSC (validation set)**:

| Class | Validation DSC | Test DSC | Target | Status |
|-------|----------------|----------|--------|--------|
| 0 (background) | 0.9996 | 0.9996 | >0.9 | PASS |
| 1 | 0.9405 | 0.9449 | >0.9 | PASS |
| 2 | 0.9544 | 0.9524 | >0.9 | PASS |
| 3 | 0.9730 | 0.9725 | >0.9 | PASS |

All four classes exceed the 0.9 DSC target on both validation and the held-out test
set. Example predictions (input / ground truth / prediction, with per-example DSC)
are visualised in `unet/outputs/unet_predictions.png`; note individual test images
vary in difficulty (e.g. one example scored 0.87 on class 1 despite the class-wide
0.94 test average) — expected variance, not a failure of the aggregate metric.
Live inference on a single test image is supported via
`python3 unet/predict.py --index N`.

## Task 3: GAN

**Architecture**: DCGAN-style Generator (5 transposed-conv layers, `1×1 → 64×64`,
BatchNorm + ReLU, `tanh` output) and Discriminator (4 conv layers, `64×64 → 1×1`,
LeakyReLU, no BatchNorm on the first layer per standard DCGAN advice). Trained at
**64×64 resolution** (smaller than VAE/UNet's 128×128) for training stability, per the
assignment's own warning about GAN convergence difficulty.

**Stability measures used**: label smoothing (real labels = 0.9), lowered Adam beta1
(0.5), a fixed noise vector for tracking the same generated images across training,
and a mode-collapse warning that monitors for Discriminator loss collapsing near 0.

**Training**: 100 epochs, `BCEWithLogitsLoss`, separate Adam optimisers for G and D
(lr 2e-4). Training took **1361.9 seconds** (~22.7 minutes) on an A100.

**Loss trajectory**: G loss fell quickly in the first ~20 epochs (9.7 → 2.5), both
losses stayed roughly balanced through epoch ~65, then G loss climbed gradually
(3.3 → 4.1) while D loss declined (0.6 → 0.4) through the remaining epochs — the
Discriminator slowly gaining ground late in training. D loss never collapsed toward 0
and showed periodic recovery spikes (epochs 66, 90, 98), indicating the adversarial
dynamic stayed active rather than fully collapsing. See `gan/outputs/gan_loss_curve.png`.

**Qualitative results**: by epoch 100, generated samples show clear cortical folding
texture, consistent skull outline and ventricle structure, and — critically — real
diversity across the 64-sample grid (varying brightness, ventricle shape, texture
density), indicating the late-training loss trend did not translate into mode
collapse. Full training progression (every 5 epochs) is saved in
`gan/outputs/samples/epoch_0001.png` through `epoch_0100.png`.

## Results summary

| Task | Metric | Result |
|------|--------|--------|
| VAE | Training time | 463.5s (30 epochs) |
| VAE | Manifold visualisation | `vae/outputs/vae_umap_latent.png`, `vae/outputs/vae_prior_samples.png` |
| UNet | Training time | 5324.0s (40 epochs) |
| UNet | Mean DSC across labels (test set) | 0.9674 (all classes individually >0.9) |
| GAN | Training time | 1361.9s (100 epochs) |
| GAN | Qualitative realism | `gan/outputs/samples/epoch_0100.png` — diverse, anatomically consistent, no mode collapse |

## Notes

- All models are implemented from scratch in PyTorch (no pretrained models used).
- Code is organised so that `data/data_loader.py` is shared across all three tasks
  to avoid duplicated data-handling logic.
