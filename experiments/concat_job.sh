#!/bin/bash
#SBATCH -J concat_csvs
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=./Sbatch-csvconcat/%A_%a.out
#SBATCH --error=./Sbatch-csvconcat/%A_%a.error

mkdir -p ./Sbatch-csvconcat

module load anaconda3/2022.05.0.1
conda activate dfldp

op_name='vsb_100'
python concat_dfs.py --op_name ${op_name}
