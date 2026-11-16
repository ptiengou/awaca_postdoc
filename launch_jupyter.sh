#!/bin/bash
echo "Requesting Slurm compute node allocation..."
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=08:00:00 --pty bash -c '
    echo "Node allocated: $HOSTNAME"
    
    # Source the correct Miniforge conda profile script
    source /net/nfs/tools/u20/Python/miniforge_24.11.3-0/etc/profile.d/conda.sh
    
    conda activate lam1env
    echo "Environment activated. Starting lightweight Jupyter server..."
    
    # Force classic notebook server backend to avoid Node/Yarn checks
    jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
'
