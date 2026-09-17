#!/bin/bash
#SBATCH --job-name=comp3710-train
#SBATCH --partition=comp3710          # confirmed: course-specific partition, A100 nodes
#SBATCH --gres=gpu:1                  # request 1 GPU (A100)
#SBATCH --cpus-per-task=8             # adjust based on DataLoader num_workers
#SBATCH --mem=32G                     # adjust based on batch size / dataset in memory
#SBATCH --time=01:00:00               # HH:MM:SS wall-clock limit - keep tight for fast iteration
#SBATCH --output=slurm-%j.out         # %j = job ID, goes in jobs/ or repo root
#SBATCH --error=slurm-%j.err

# --- Environment setup ---
# No python module needed - system python3 (3.11.13) is used directly.
# No conda available either - using a plain venv instead.
module load cuda/12.2

source ~/comp3710-env/bin/activate

# --- Sanity check: confirm GPU is visible ---
nvidia-smi

# --- Run training ---
# Edit this to point at whichever task's train.py you're running
python3 vae/train.py

echo "Job finished with exit code $?"
