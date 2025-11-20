#!/bin/bash
#SBATCH --job-name=ames_combine
#SBATCH --output=logs/ames-combine_%j.out
#SBATCH --error=logs/ames-combine_%j.err
#SBATCH --partition=STD-s-96h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source /lustre/scratch/soarjame-bachelor-thesis/envs/mlenv/bin/activate
cd /lustre/scratch/soarjame-bachelor-thesis/code/Boruta_Main_Amplitude

echo "Combining scenario results on host $(hostname)"
srun python combine_ames_results.py
