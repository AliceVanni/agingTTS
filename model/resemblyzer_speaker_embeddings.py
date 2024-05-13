from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import os
import json
import torch

corpus_folder = 'agingTTS'

with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
    speakers = json.load(f)
    n_speaker = len(speakers)

speaker_embeddings = {}

for i, speaker in enumerate(os.listdir(corpus_folder)):
    speaker_files = list(Path(corpus_folder).glob(f'{speaker}/*.wav'))
    if speaker_files:
        wav = np.concatenate([preprocess_wav(file) for file in speaker_files])
    else:
        print(f'No .wav files found for speaker {speaker}')
        continue

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    speaker_embeddings[speaker] = torch.from_numpy(embed).float()

torch.save(speaker_embeddings, 'speaker_emb_from_resemblyzer.pt')