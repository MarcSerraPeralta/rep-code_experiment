#!/bin/bash -l
#
#SBATCH --job-name="rep"
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=m.serraperalta@tudelft.nl
#SBATCH --mail-type=END,FAIL
#SBATCH --output=job_outputs/train_qrennd.%j.out
#SBATCH --error=job_outputs/train_qrennd.%j.err

module load 2022r2
module load python

export ENV_NAME="rep_code"
source /scratch/${USER}/virtual_environments_gpu/${ENV_NAME}/bin/activate

module load openmpi
module load py-tensorflow

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun --mpi=pmix python train.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

deactivate