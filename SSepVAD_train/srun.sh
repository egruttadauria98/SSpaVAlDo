#!/bin/bash

#SBATCH --output=009_train_conv3_%j.out
#SBATCH --error=009_train_conv3_%j.err
#SBATCH --time=70:00:00
#SBATCH --partition=A40
#SBATCH --gpus=1

set -x
srun python -u train.py --conf_id "009"
###srun python -u proof_better_model.py
