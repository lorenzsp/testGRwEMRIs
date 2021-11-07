#!/bin/bash -l
# Standard output and error:
#SBATCH -o mig.out.%j
#SBATCH -e mig.err.%j
# Initial working directory:
#SBATCH -D ./
#
#SBATCH -J scalar
#
# Node feature:
#SBATCH --constraint="gpu"
# Specify type and number of GPUs to use:
#   GPU type can be v100 or rtx5000
#SBATCH --gres=gpu:v100:2         # If using both GPUs of a node
# #SBATCH --gres=gpu:v100:1       # If using only 1 GPU of a shared node
# #SBATCH --mem=92500             # Memory is necessary if using only 1 GPU
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # If using both GPUs of a node
# #SBATCH --ntasks-per-node=20    # If using only 1 GPU of a shared node
#
# #SBATCH --mail-type=none
# #SBATCH --mail-user=userid@example.mpg.de
#
# wall clock limit:
#SBATCH --time=24:00:00

module load gcc cuda
module load anaconda
conda activate scalar_few_env

# # dev run
# # srun --time=00:05:00 --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --partition=gpudev --gres=gpu:v100:2 ./emri_example.py

# Run the program:
python run.py