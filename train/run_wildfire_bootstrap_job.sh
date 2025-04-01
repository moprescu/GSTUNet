#!/bin/bash
#SBATCH --job-name=gstunet_wildfire_bootstrap           # Job name
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -A m1266
#SBATCH --output=.unet_out/%x_%j.out  # Redirect stdout to folder unet_out/
#SBATCH --error=.unet_out/%x_%j.err   # Redirect stderr to folder unet_out/

# Load your software stack
module load pytorch

# Run the Python script
python wildfire_experiment_bootstrap.py --random_seed "${RANDOM_SEED}"
python wildfire_experiment_bootstrap_unet.py --random_seed "${RANDOM_SEED}"
python wildfire_experiment_bootstrap_stcinet.py --random_seed "${RANDOM_SEED}"
