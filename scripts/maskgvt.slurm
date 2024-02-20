#!/bin/bash -l

#SBATCH -A collabrobogroup
#SBATCH --array=0
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH -N 1
#SBATCH --cpus-per-task=50
#SBATCH -J official_runs

# #SBATCH --ntasks-per-node=1
# #SBATCH --begin=2023-10-29T3:00:00
# -C gpupod is included to only use A100s with 80 GB
# #SBATCH -C gpupod

#SBATCH --array=4,8,16,32
#SBATCH --output=logs/maskgvt-log-%A-%a.log

module purge
module load apptainer

export MASKGVT_BATCH_SIZE=$SLURM_ARRAY_TASK_ID
export MASKGVT_WORKDIR=workdir/maskgvt/$SLURM_ARRAY_TASK_ID
mkdir -p $MASKGVT_WORKDIR
bash scripts/maskgvt.sh
