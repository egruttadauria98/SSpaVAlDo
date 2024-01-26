#!/bin/bash

#SBATCH --output=012_online_%j.out
#SBATCH --error=012_online_%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1

set -x
srun python -u online_inference.py --conf_id "012"
