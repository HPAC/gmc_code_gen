#!/bin/bash
#SBATCH -A {ACCOUNT_ID}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -c 1
#SBATCH --time=00:15:00
#SBATCH --error=job_output/compilation.%J.err
#SBATCH --output=job_output/compilation.%J.out

module purge > /dev/null 2>&1

# The following are modules to be loaded in the environment:
#   - GCC with support for C++17
#   - CMake > 3.15
#   - OpenBLAS (if possible 0.3.27)
# Make sure to also load OpenMP separately if it does not come with GCC.
module load GCC/13.3.0
module load CMake
module load OpenBLAS/0.3.27

cmake -DCMAKE_BUILD_TYPE=Release -S .. -B .
cmake --build .
