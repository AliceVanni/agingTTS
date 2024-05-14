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

from pytorch_revgrad import RevGrad

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


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
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

            self.latent_speaker_emb = nn.Linear(
                model_config["transformer"]["encoder_hidden"],
                model_config["transformer"]["encoder_hidden"],
            )

        self.age_classifier = nn.Linear(
                model_config["transformer"]["encoder_hidden"], 3,
            )
        
        self.age_emb = nn.Embedding(3, model_config["transformer"]["encoder_hidden"])
        self.revgrad = RevGrad()
    
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
        
        age_embeddings = self.age_emb(ages)
        #print(f'Age embedding calculated in training: {age_embeddings}')
        
        # The age embedding is added to the input tensor
        if age_embeddings is not None:
            output = output + age_embeddings.unsqueeze(1).expand(-1, max_src_len, -1)
        
        # The speaker embedding is also added to the output
        if self.speaker_emb is not None:
            speaker_embedding = self.speaker_emb(speakers)
            latent_speaker_embedding = self.latent_speaker_emb(speaker_embedding)
            output = output + self.revgrad(latent_speaker_embedding).unsqueeze(1).expand(
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
