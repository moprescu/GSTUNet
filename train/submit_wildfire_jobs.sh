#!/bin/bash

# 1) Seed values for bootstrapping
seed_values=($(seq 1 40))
# 2) Define the LOCAL flag
LOCAL=true  # Set to true to run jobs locally, or false to submit via sbatch

# 3) Next step: For each beta_1 and dim_horizon, submit the training job
echo "Submitting SLURM jobs..."
for s in "${seed_values[@]}"
do
  if [ "$LOCAL" = true ]; then
    # Run jobs locally
    echo "Running run_wildfire_bootstrap_job.sh locally for random_seed=$s..."
    RANDOM_SEED=$s ./run_wildfire_bootstrap_job.sh
  else
    # Submit jobs to SLURM
    echo "Submitting wildfire bootstrap job for random_seed=$s via sbatch..."
    sbatch --export=RANDOM_SEED=$s run_wildfire_bootstrap_job.sh
  fi
done