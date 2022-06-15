#!/bin/bash
#SBATCH -J check_device_gpu_sdummont
#SBATCH --nodes=1
#SBATCH -p ict_cpu
#SBATCH -A brcluster

python check_device_parallel.py