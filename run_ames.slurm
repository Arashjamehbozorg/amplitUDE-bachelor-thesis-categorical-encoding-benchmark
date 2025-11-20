#!/bin/bash
#SBATCH --job-name=ames_scenarios
#SBATCH --output=logs/ames-scen_%A_%a.out
#SBATCH --error=logs/ames-scen_%A_%a.err
#SBATCH --partition=STD-s-96h

#SBATCH --array=0-3             # scenarios 0,1,2,3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112     # use all CPU cores on a STD node
#SBATCH --mem=400G              # well within 512G per node
#SBATCH --time=24:00:00         # enough for heavy scenario 2/4

mkdir -p logs

# Respect Slurm CPU allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TBB_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}

source /lustre/scratch/soarjame-bachelor-thesis/envs/mlenv/bin/activate
cd /lustre/scratch/soarjame-bachelor-thesis/code/Boruta_Main_Amplitude

echo "Running scenario ${SLURM_ARRAY_TASK_ID} on host $(hostname)"

srun python encoding_benchmark_defs.py --scenario-id ${SLURM_ARRAY_TASK_ID}
