# # Remember to change the interpreter in the code
module load cuda
conda activate ls_few_env
srun --time=00:15:00 --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --partition=gpudev --gres=gpu:v100:1 python run.py
