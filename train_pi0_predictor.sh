#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=72GB
#SBATCH --job-name=train_predictor_v7
#SBATCH --output=slurm-output/slurm-train_predictor_v7-%j.out
#SBATCH --gpus-per-node=a100:2

source $HOME/.bashrc
module load Miniconda3/22.11.1-1

conda activate openpi

python scripts/train.py pi0_libero_predictor --exp-name=predictor_v7 --overwrite