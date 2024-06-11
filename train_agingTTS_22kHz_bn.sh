#!/bin/bash

#SBATCH --gpus-per-node=v100:1
#SBATCH --time=18:00:00
#SBATCH --mem=124GB

source /home2/s5298873/venvs/agingTTS_venv/bin/activate
python3 train_bn.py --restore_step 180000 -p config/agingTTS_22kHz_bn/preprocess.yaml -m config/agingTTS_22kHz_bn/model.yaml -t config/agingTTS_22kHz_bn/train.yaml