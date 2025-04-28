#!/bin/bash
#SBATCH -J df2vslr
#SBATCH --array=1-100
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=./Sbatch-df2lr-100/%A_%a.out
#SBATCH --error=./Sbatch-df2lr-100/%A_%a.error

mkdir -p ./Sbatch-df2lr-100

module load anaconda3/2022.05.0.1
conda activate dfldp

n=$SLURM_ARRAY_TASK_ID
iteration=$(sed -n "${n} p" eps_lambda.csv)  # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

num_runs=100
B=1000
eps=$(echo ${iteration} | cut -d "," -f 1)
lambda_dfl=$(echo ${iteration} | cut -d "," -f 2)
lambda_lr=$(echo ${iteration} | cut -d "," -f 3)

python dfl2_vs_lr_bestc_sbatch.py --batch_size ${B} --num_runs ${num_runs} --epsilon ${eps} --lamb_dfl ${lambda_dfl} --lamb_lr ${lambda_lr}
