#!/bin/bash

#SBATCH --gpus-per-node=v100:1
#SBATCH --time=20:00:00
#SBATCH --mem=124GB

source /home2/s5298873/venvs/agingTTS_venv/bin/activate
python3 train.py --restore_step 140000 -p config/agingTTS_22kHz/preprocess.yaml -m config/agingTTS_22kHz/model.yaml -t config/agingTTS_22kHz/train.yaml