#!/bin/bash
#SBATCH -A {YOUR_ACCOUNT}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -N 1
#SBATCH -c {MAX_NUM_CORES_IN_NODE}
#SBATCH --time=15:00:00
#SBATCH --error=job_output/generate_times.%J.err
#SBATCH --output=job_output/generate_times.%J.out
#SBATCH --array=0-999

# This script assumes each computing node has one processor. If computing 
# nodes in your cluster have more than one and using multiple processors 
# simultaneously is safe regarding the measured execution time, adapt it to 
# run multiple things at the same time.

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job.
module load GCCcore/13.3.0
module load CMake
module load OpenBLAS/0.3.27
module load hwloc/2.10.0 # for hwloc-bind

export OMP_NUM_THREADS=14 # set number of cores
export OMP_PROC_BIND=true # logical threads bound to cores

# Specify the node (in this case processor) where the threads will be executed 
# with hwloc-bind. Use --membind to fix where the memory used by the process
# should reside.

expr_ID="${SLURM_ARRAY_TASK_ID}"
N_s=1000
reps=20

# Arguments   : <exprID> <n> <N_s> <reps> <expr_file> <out_file>
# <expr_ID>   : number identifying which expression in <expr_file> is used as input
# <n>         : length of the chain
# <N_s>       : number of instances to randomly sample
# <reps>      : times each pair (alg,instance) is executed
# <expr_file> : path relative to working directory. File with a collection of expressions
# <out_file>  : path relative to working directory. File where result timings are dumped
hwloc-bind --cpubind socket:0 --membind node:0 -- ./experiments/timings_10comb ${expr_ID} 7 ${N_s} ${reps} tmp/expr_exp2.txt times/vars/mc7_${expr_ID}.txt
