#!/bin/bash

#SBATCH --output=002_train_%j.out
#SBATCH --error=002_train_%j.err
#SBATCH --time=100:00:00
#SBATCH --partition=A40
#SBATCH --gpus=1

set -x
srun python -u train.py --conf_id "002"
