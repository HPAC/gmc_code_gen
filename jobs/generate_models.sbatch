#!/bin/bash
#SBATCH -A {YOUR_ACCOUNT}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -N 1
#SBATCH -c {MAX_NUM_CORES_IN_NODE}
#SBATCH --time=02:00:00
#SBATCH --error=job_output/model_generation.%J.err 
#SBATCH --output=job_output/model_generation.%J.out

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job.
module load GCCcore/13.3.0
module load CMake
module load OpenBLAS/0.3.27
module load hwloc/2.10.0 # for hwloc-bind

export OMP_NUM_THREADS=14 # set number of cores
export OMP_PROC_BIND=true # logical threads bound to cores

# Specify the node (node as per lstopo/hwloc terminology -- a processor) where 
# the threads will be executed with hwloc-bind. Use --membind to fix where the 
# memory used by the process should reside.

# args: <n_reps>
# <n_reps>: number of times each kernel is executed on each instance.
hwloc-bind --cpubind socket:0 --membind node:0 -- ./experiment/generate_models 20
