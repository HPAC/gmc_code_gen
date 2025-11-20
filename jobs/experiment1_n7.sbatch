#!/bin/bash
#SBATCH -A {YOUR_ACCOUNT}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -c 1
#SBATCH --time=15:00:00
#SBATCH --error=job_output/experiment1_n7/job.%J.err
#SBATCH --output=job_output/experiment1_n7/job.%J.out
#SBATCH --array=0-999

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job.
module load GCC/13.3.0
module load CMake
module load OpenBLAS/0.3.27

n=7
# args: <n> <N_t> <N_v> <procID> <n_proc> <base_file_out>
#  <n>:             length of the chain.
#  <N_t>:           number of instances in training.
#  <N_v>:           number of instances in validation.
#  <procID>:        process id.
#  <n_proc>:        total number of processes. Together with the previous arg 
#                   determine the range of generation.
#  <fname_out>:     path where results are dumped. Relative to working dir.
./experiments/experiment1 ${n} 100000 1000 ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT} ../results/experiment1/n${n}/
