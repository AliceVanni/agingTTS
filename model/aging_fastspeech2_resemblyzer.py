''' AGING FASTSPEECH2:
    Implementation of FastSpeech2 with age embeddings in order to  be able to 
    control the (perceived) age of the synthesised voice.
    
    The implementation is based on https://github.com/ming024/FastSpeech2.
'''

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from collections import OrderedDict
from utils.tools import get_mask_from_lengths
from pathlib import Path 


class AgingFastSpeech2(nn.Module):
    """ FastSpeech2 with age control"""

    def __init__(self, preprocess_config, model_config):
        super(AgingFastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            speaker_emb_dict = torch.load(f'./{model_config["speaker_embedding"]["pretrained_speaker_embeddings"]}').float()
            self.speaker_emb = nn.Embedding.from_pretrained(
                                            speaker_emb_dict, freeze=True)
            
        if model_config["multi_age"]: 
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "ages.json"), "r") as f:
                n_age = len(json.load(f))
            self.age_emb = nn.Embedding(n_age, model_config["age_embedding"]["embedding_hidden"])
            self.age_proj = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1",
                        nn.Linear(model_config["age_embedding"]["embedding_hidden"], model_config["transformer"]["encoder_hidden"]),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("linear_2", nn.Linear(model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"])),
                ]
            )
        )
        
    def forward(
        self,
        speakers,
        ages,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
            
        # The encoder processes the input text and it is added to the encoder output
        output = self.encoder(texts, src_masks)
        
        # The age embedding is added to the input tensor
        if self.age_emb is not None:
            output = output + self.age_proj(self.age_emb(ages)).unsqueeze(1).expand(-1, max_src_len, -1)
            #output = output + self.age_emb(ages).unsqueeze(1).expand(-1, max_src_len, -1)
        
        # The speaker embedding is also added to the output
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )           

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )