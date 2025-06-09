#!/bin/bash

#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Stdout output (%j = Job ID)
#SBATCH -e job.%j.err         # Stderr output
#SBATCH -N 1                  # Total number of nodes
#SBATCH -n 4                  # Total number of tasks
##SBATCH --ntasks-per-node=8
#SBATCH -t 4:00:00           # Time limit
##SBATCH -p mi3008x            # Partition name
#SBATCH -p mi2508x
##SBATCH -p mi1008x
##SBATCH -p mi2104x
# Load necessary modules (customize as needed)
module load pytorch
source $WORK/myenv/bin/activate

module load pytorch

echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"


echo "Launching 4 jobs on different GPUs..."
# Replace this with the actual Python training script path
#cd DT_QGT
python_script="atari/run_QGT.py"

# Get master node hostname from SLURM
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500  # or a port of your choice

# Fallback check
if [ -z "$MASTER_ADDR" ]; then
    echo "MASTER_ADDR not set! SLURM_JOB_NODELIST was empty."
    exit 1
fi


# Run your Python script using torchrun or mpirun depending on your use case
# For PyTorch DDP
#torchrun --nproc_per_node=4 --nnodes=1 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500  $python_script


python -c "import h5py; print('h5py found')"



# Activate the correct environment for all torchrun workers
VENV_PATH="/work1/javidi/mahdis/myenv"
export PATH="$VENV_PATH/bin:$PATH"
export PYTHONPATH="$VENV_PATH/lib/python3.9/site-packages:$PYTHONPATH"
export VIRTUAL_ENV="$VENV_PATH"

torchrun \
  --nproc_per_node=4 \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
 	$python_script  \
--epochs 20 --batch_size 512 \
--learning_rate 4e-3 \
--criterion "bce" \
--seed 678 \
--label_smoothing 0 \
--dataset_path data/k_10/10M_k10.h5 \
--k 10 \
#--block_size 20
#--resume_ckpt_path models/dulcet-field-88.pth \
#--no_lr_decay
#--repeated_dataset


echo "Done"
# OR, for single-node testing, simply:
# python your_script.py
