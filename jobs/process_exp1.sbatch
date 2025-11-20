#SBATCH -A {YOUR_ACCOUNT}
#SBATCH -C {TARGET_COMPUTING_NODE}
#SBATCH -c {NUMBER_OF_CORES}
#SBATCH --time=02:00:00
#SBATCH --error=job_output/join_bin_files.%J.err
#SBATCH --output=job_output/join_bin_files.%J.out

module purge > /dev/null 2>&1

# Make sure to load here the corresponding environment so that Python and Numpy
# are available.
module load GCC/12.3.0
module load SciPy-bundle/2023.07

python ../plotting/processing_exp1.py