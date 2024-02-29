# Define different sets of (M, mu, a, e0)
parameter_sets=(
    "1e5 1e1 0.95 0.4 2.0"
)

# Loop through each parameter set
for parameters in "${parameter_sets[@]}"; do
    # Split parameters into individual variables
    IFS=' ' read -r M mu a e0 Tobs<<< "$parameters"

    # Set other variables here
    dt=10.0
    p0=13.0
    x0=1.0
    charge=0.0
    nwalkers=32
    ntemps=2
    nsteps=500000
    dev=0

    # Submit Slurm job for this combination of variables
    sbatch <<EOF
#!/bin/bash -l
#SBATCH -o ./job_M_${M}_mu_${mu}_a_${a}_e0_${e0}_Tobs_${Tobs}.out.%j
#SBATCH -e ./job_M_${M}_mu_${mu}_a_${a}_e0_${e0}_Tobs_${Tobs}.err.%j
#SBATCH -J M_${M}_mu_${mu}_a_${a}_e0_${e0}_Tobs_${Tobs}
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=125000
#SBATCH --time=24:00:00
#SBATCH --priority=10000

module purge
module load cuda/12.1
eval \$(conda shell.bash hook)
conda activate bgr_env

export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

python mcmc.py -Tobs $Tobs -dt $dt -M $M -mu $mu -a $a -p0 $p0 -e0 $e0 -x0 $x0 -charge $charge -dev $dev -nwalkers $nwalkers -ntemps $ntemps -nsteps $nsteps
EOF
done
