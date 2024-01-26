#!/bin/bash

#SBATCH --output=0_train_007-012_%j.out
#SBATCH --error=0_train_007-012_%j.err
#SBATCH --time=140:00:00
#SBATCH --partition=audible
#SBATCH --gpus=1

set -x
srun python -u train.py --conf_id "007" > eval007_output.log 2> eval007_error.log &
srun python -u train.py --conf_id "008" > eval008_output.log 2> eval008_error.log &
srun python -u train.py --conf_id "009" > eval009_output.log 2> eval009_error.log &
srun python -u train.py --conf_id "010" > eval010_output.log 2> eval010_error.log &
srun python -u train.py --conf_id "011" > eval011_output.log 2> eval011_error.log &
srun python -u train.py --conf_id "012" > eval012_output.log 2> eval012_error.log &
wait
