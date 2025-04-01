#!/bin/bash
#SBATCH --job-name=unet_train           # Job name
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH -t 00:10:00
#SBATCH -A m1266
#SBATCH --output=.unet_out/%x_%j.out  # Redirect stdout to folder unet_out/
#SBATCH --error=.unet_out/%x_%j.err   # Redirect stderr to folder unet_out/

#############################
# Load your software stack #
#############################
#module load pytorch

# Debug info
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "beta_1 = ${BETA_1}"
echo "dim_horizon = ${DIM_HORIZON}"

#######################################
# Run the Python script with srun     #
#######################################
python train_stcinet.py \
  --beta_1 "${BETA_1}" \
  --dim_horizon "${DIM_HORIZON}" \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --num_epochs 100 \
  --early_stopping_patience 10 \
  --scheduler_patience 5 
