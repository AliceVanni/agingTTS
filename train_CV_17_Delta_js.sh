#!/bin/bash

#SBATCH --gpus-per-node=v100:1
#SBATCH --time=05:30:00

source $HOME/venvs/FS_venv/bin/activate
python3 train.py --restore_step 3000 -p config/CV_17_Delta/preprocess.yaml -m config/CV_17_Delta/model.yaml -t config/CV_17_Delta/train.yaml
deactivate