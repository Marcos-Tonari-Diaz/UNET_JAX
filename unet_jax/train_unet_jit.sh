#!/usr/bin/bash
#SBATCH --account=brcluster
#SBATCH --partition=ict_gpu
#SBATCH --job-name=unet_jax_jit    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m221525@dac.unicamp.br     # Where to send mail      
#SBATCH --output=unet_jax_jit_%j.log   # Standard output and error log

python train_unet_jit.py
