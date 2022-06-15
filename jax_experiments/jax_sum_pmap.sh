#!/bin/bash
#SBATCH -J jax_sum_pmap_4nodes
#SBATCH -p ict_gpu
#SBATCH --nodes=4
#SBATCH -A brcluster
#SBATCH --output=results/%x.%j.out
#SBATCH --error=results/%x.%j.err

srun python jax_sum_pmap.py