#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=eb-4
#SBATCH --output=%j.out

module load miniconda/3
conda activate sheafnn

python torus-4.py