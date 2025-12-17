#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=42GB
#SBATCH --job-name=finetune_predictor_vae
#SBATCH --output=slurm-output/slurm-finetune_predictor_vae-%j.out
#SBATCH --gpus-per-node=a100:1

source $HOME/.bashrc
module load Miniconda3/22.11.1-1

conda activate pipredictor

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_predictor.py pi0_libero_predictor --exp-name=predictor_fine_tune --overwrite