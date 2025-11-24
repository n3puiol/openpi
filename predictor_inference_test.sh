#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=predictor_inference_test
#SBATCH --output=slurm-output/slurm-predictor_inference_test-%j.out
#SBATCH --gpus-per-node=a100:1

source $HOME/.bashrc
module load Miniconda3/22.11.1-1

conda activate openpi

python scripts/predictor_inference.py
