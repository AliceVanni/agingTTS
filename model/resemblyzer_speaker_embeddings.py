from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

import numpy as np

import os
import argparse
import json
import torch
import yaml

def resemblyzer_speaker_embedding(args):

    '''
    Training of resemblyzer speaker encoder on custom dataset.
    
    Returns the matrix with the speaker embeddings.
    '''
    
    corpus_folder = args.corpus_folder
    preprocess_config = yaml.load(
            open(f'config/{corpus_folder}/preprocess.yaml', "r"), Loader=yaml.FullLoader
        )
    model_config = yaml.load(
            open(f'config/{corpus_folder}/model.yaml', "r"), Loader=yaml.FullLoader
        )
    
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
        speakers = json.load(f)
        n_speaker = len(speakers)
    
    speaker_embedding_list = []
    
    for i, speaker in enumerate(os.listdir(corpus_folder)):
        speaker_files = list(Path(corpus_folder).glob(f'{speaker}/*.wav'))
        if speaker_files:
            wav = np.concatenate([preprocess_wav(file) for file in speaker_files])
        else:
            print(f'No .wav files found for speaker {speaker}')
            continue
    
        encoder = VoiceEncoder()
        embed = encoder.embed_utterance(wav)
        tensor_embed = torch.from_numpy(embed)
        speaker_embedding_list.append(tensor_embed)
    
    speaker_embedding_matrix = torch.stack(speaker_embedding_list)
    
    print(f'Speaker embeddings matrix shaped {speaker_embedding_matrix.shape}')
    
    output_filename = args.output_file
    torch.save(speaker_embedding_matrix, model_config["speaker_embedding"]["pretrained_speaker_embeddings"])
    print(f"Pretrained embeddings saved in {model_config["pretrained_speaker_embeddings"]}")
    
    return speaker_embedding_matrix
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--corpus_folder",
        type=str,
        required=True,
        help="Name of the corpus folder",
    )
    args = parser.parse_args()
    
    resemblyzer_speaker_embedding(args)