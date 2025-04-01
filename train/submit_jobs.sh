#!/bin/bash

# 1) Create arrays of beta_1 and dim_horizon values
beta_1_values=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
dim_horizon_values=(5 10)
#beta_1_values=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
#dim_horizon_values=(2 5 10)

#module load pytorch

# Define the LOCAL flag
LOCAL=true  # Set to true to run jobs locally, or false to submit via sbatch

# 2) First step: run dgp_linear.py with all beta_1 values at once
#    We will pass them as a space-separated list to the script.
echo "Running dgp_linear.py to generate data for all beta_1 values..."

# Convert the array into a single string, e.g. "0.0 0.5 1.0 1.5 2.0"
beta_1_list_str="${beta_1_values[@]}"
echo "beta_1_list_str = $beta_1_list_str"

# Run dgp_linear.py with a command-line argument that is a space-separated list
python ../data/dgp_linear.py --beta_1 $beta_1_list_str --output_dir ../data/simulated_data/linear --mean_path

# 3) Next step: For each beta_1 and dim_horizon, submit the training job
echo "Submitting SLURM jobs..."
for b1 in "${beta_1_values[@]}"
do
  for dh in "${dim_horizon_values[@]}"
  do
    if [ "$LOCAL" = true ]; then
      # Run jobs locally
      echo "Running run_unet_job.sh locally for beta_1=$b1, dim_horizon=$dh..."
      BETA_1=$b1 DIM_HORIZON=$dh ./run_unet_job.sh
      echo "Running run_gstunet_job.sh locally for beta_1=$b1, dim_horizon=$dh..."
      BETA_1=$b1 DIM_HORIZON=$dh ./run_gstunet_job.sh
      echo "Running run_stcinet_job.sh locally for beta_1=$b1, dim_horizon=$dh..."
      BETA_1=$b1 DIM_HORIZON=$dh ./run_stcinet_job.sh
      echo "Running run_ipw_job.sh locally for beta_1=$b1, dim_horizon=$dh..."
      BETA_1=$b1 DIM_HORIZON=$dh ./run_ipw_job.sh
    else
      # Submit jobs to SLURM
      echo "Submitting unet job for beta_1=$b1, dim_horizon=$dh via sbatch..."
      sbatch --export=BETA_1=$b1,DIM_HORIZON=$dh run_unet_job.sh
      echo "Submitting gstunet job for beta_1=$b1, dim_horizon=$dh..."
      sbatch --export=BETA_1=$b1,DIM_HORIZON=$dh run_gstunet_job.sh
      echo "Submitting stcinet job for beta_1=$b1, dim_horizon=$dh via sbatch..."
      sbatch --export=BETA_1=$b1,DIM_HORIZON=$dh run_stcinet_job.sh
      echo "Submitting ipw job for beta_1=$b1, dim_horizon=$dh via sbatch..."
      sbatch --export=BETA_1=$b1,DIM_HORIZON=$dh run_ipw_job.sh
    fi
  done
done

if [ "$LOCAL" = true ]; then
  cd ../data/simulated_data && python summarize.py
fi

