#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH --output=logs/log-%A-%a.log
#SBATCH -J official_runs

# #SBATCH --ntasks-per-node=1
# #SBATCH --begin=2023-10-29T3:00:00
# -C gpupod is included to only use A100s with 80 GB
# #SBATCH -C gpupod

module purge
module load apptainer

bash mini_vqgan.sh
