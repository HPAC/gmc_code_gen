#!/bin/bash
#SBATCH -A {YOUR_ACCOUNT}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -c 1
#SBATCH --time=15:00:00
#SBATCH --error=job_output/experiment_2/job.%J.err
#SBATCH --output=job_output/experiment_2/job.%J.out

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job.
module load GCC/13.3.0
module load CMake
module load OpenBLAS/0.3.27

N_t=100000
n_shapes=1000

# args: <N_t> <n_shapes> <base_fname_time> <base_fname_out>
#  <N_t>:                    number of instances in training
#  <n_shapes>:               number of shapes in the dir
#  <base_fname_times_vars>:  base path and name for files with timings
#  <base_fname_times_arma>:  base path and name for files with timings in Armadillo
#  <base_fname_out>:         base path and name for output files. Two files are 
#                            created: one with the results training with FLOPs,
#                            another, with estimations via models
./experiments/experiment2 ${N_t} ${n_shapes} ../times/vars/mc7_ ../times/arma/mc7_ ../results/experiment2/