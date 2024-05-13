from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from tqdm import tqdm

import numpy as np

import os
import json
import torch
import yaml

corpus_folder = 'agingTTS'
preprocess_config = yaml.load(
        open(f'config/{corpus_folder}/preprocess.yaml', "r"), Loader=yaml.FullLoader
    )

with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
    speakers = json.load(f)
    n_speaker = len(speakers)

speaker_embedding_matrix = torch.empty(256) # I assume the speaker embeddings will always be 256, but I should find a way to not hard code the tensor dimension

for i, speaker in tqdm(enumerate(os.listdir(corpus_folder))):
    speaker_files = list(Path(corpus_folder).glob(f'{speaker}/*.wav'))
    if speaker_files:
        wav = np.concatenate([preprocess_wav(file) for file in speaker_files])
    else:
        print(f'No .wav files found for speaker {speaker}')
        continue

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    tensor_embed = torch.from_numpy(embed)
    speaker_embedding_matrix = torch.cat((speaker_embedding_matrix, tensor_embed))

print(f'Speaker embeddings matrix: {speaker_embedding_matrix[1:]}')
torch.save(speaker_embedding_matrix[1:], 'speaker_emb_from_resemblyzer.pt')