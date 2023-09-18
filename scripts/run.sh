#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=1-torus-full
#SBATCH --output=%j.out

module load miniconda/3
conda activate sheafnn

python src/torus.py