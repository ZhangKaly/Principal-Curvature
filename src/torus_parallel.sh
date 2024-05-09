#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=torus-full
#SBATCH --output=%j.out

module load miniconda/3
conda activate sheafnn


sbatch job_torus.sh