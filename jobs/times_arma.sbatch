#!/bin/bash
#SBATCH -A {YOUR_ACCOUNT}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -N 1
#SBATCH -c {MAX_NUM_CORES_IN_NODE}
#SBATCH --time=01:00:00
#SBATCH --error=job_output/times_arma.%J.err
#SBATCH --output=job_output/times_arma.%J.out
#SBATCH --array=0-999

# This script assumes each computing node has one processor. If computing 
# nodes in your cluster have more than one and using multiple processors 
# simultaneously is safe regarding the measured execution time, adapt it to 
# run multiple things at the same time.

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job.
module load GCC/13.3.0
module load CMake
module load OpenBLAS/0.3.27
module load hwloc/2.10.0 # for hwloc-bind

export OMP_NUM_THREADS=14 # set number of cores
export OMP_PROC_BIND=true # logical threads bound to cores

# Specify the node (in this case processor) where the threads will be executed 
# with hwloc-bind. Use --membind to fix where the memory used by the process
# should reside.

expr_ID="${SLURM_ARRAY_TASK_ID}"
reps=20

# Arguments          : <exprID> <n> <N_s> <reps> <expr_file> <out_file>
# <expr_ID>          : number identifying which expression in <expr_file> is used as input
# <reps>             : times each pair (alg,instance) is executed
# <fname_times_vars> : path relative to working directory. File with the timings of variants. Used to read the instances generated
# <out_file>         : path relative to working directory. File where result timings are dumped
hwloc-bind --cpubind socket:0 --membind node:0 -- ./experiments/experiment_arma ${expr_ID} ${reps} times/vars/mc7_${expr_ID}.txt times/arma/mc7_${expr_ID}.txt 
