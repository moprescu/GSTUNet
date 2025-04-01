#!/bin/bash
#SBATCH --job-name=gstunet_train        # Job name
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH -t 00:30:00
#SBATCH -A m1266
#SBATCH --output=.gstunet_out/%x_%j.out  # Redirect stdout to .gstunet_out/
#SBATCH --error=.gstunet_out/%x_%j.err   # Redirect stderr to .gstunet_out/

##################################
# Load the required software     #
##################################
#module load pytorch

# Debug info: Check environment variables passed in
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "beta_1 = ${BETA_1}"
echo "dim_horizon = ${DIM_HORIZON}"

if [ "${DIM_HORIZON}" -le 5 ]; then
  NUM_EPOCHS_WARM=1
else
  NUM_EPOCHS_WARM=7
fi

# Uncomment for curriculum ablation testing
#NUM_EPOCHS_WARM=0

###########################################
# Run the Python script with srun + flags #
###########################################
python train_gstunet.py \
    --beta_1 "${BETA_1}" \
    --dim_horizon "${DIM_HORIZON}" \
    --batch_size 4 \
    --learning_rate 0.0005 \
    --learning_rate_warm 0.0005 \
    --num_epochs 100 \
    --num_epochs_warm "${NUM_EPOCHS_WARM}" \
    --early_stopping_patience 10 \
    --scheduler_patience 3 \
    --h_size 16 \
    --fc_layers 8 \
    --use_attention