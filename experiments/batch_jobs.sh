#!/bin/bash
#SBATCH -J dflr_vlargeeps
#SBATCH --array=1-300
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=./Sbatch-vlargeeps-300/%A_%a.out
#SBATCH --error=./Sbatch-vlargeeps-300/%A_%a.error

mkdir -p ./Sbatch-vlargeeps-300

module load anaconda3/2022.05.0.1
conda activate dfldp

n=$SLURM_ARRAY_TASK_ID
iteration=$(sed -n "${n}p" eps_vlarge.csv)  # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

num_runs=100
B=1000
eps=$(echo ${iteration} | cut -d "," -f 1)
lambda_dfl=$(echo ${iteration} | cut -d "," -f 2)
lambda_lr=$(echo ${iteration} | cut -d "," -f 3)

# Comma-separated string of c_dfl values
c_dfl_values="0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,8,10"
output_dir='vlargeeps'

python dfl2_vs_lr_bestc_sbatch.py \
    --batch_size ${B} \
    --num_runs ${num_runs} \
    --epsilon ${eps} \
    --lamb_dfl ${lambda_dfl} \
    --lamb_lr ${lambda_lr} \
    --c_dfl_values "${c_dfl_values}" \
    --output_dir ${output_dir}