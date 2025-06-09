#!/bin/bash

#SBATCH -J data_gen_with_bounds        # Job name
#SBATCH -o data_gen.%j.out    # Standard output log (%j expands to jobId)
#SBATCH -e data_gen.%j.err    # Standard error log
#SBATCH -N 1               # Number of nodes
#SBATCH -n 4               # Number of tasks (4 different learning rates)
#SBATCH -t 05:00:00        # Maximum run time (hh:mm:ss)
#SBATCH -p mi2104x         # Partition name
##SBATCH -p mi2508x         # Partition name
##SBATCH --cpus-per-task=30  # CPUs per task (adjust as needed)
##SBATCH --mem=1M
##SBATCH --mem-per-cpu=4G
source $WORK/myenv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


# Optional: log environment for debugging
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

echo "START"

python data_gen.py --n_cores 128 \
--num_samples 10000000 \
--file_name data/k_10/10M \
--k 10
echo "end"


