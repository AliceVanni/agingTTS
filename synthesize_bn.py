# Doing inference with FastSpeech2 with age control

import re
import argparse
from string import punctuation

import torch
import yaml
import json
import numpy as np
from torch.utils.data import DataLoader
#from g2p_en import G2p
import eng_to_ipa as ipa

from utils.model_bn import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, text_to_sequence_ipa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    #g2p = G2p()
    
    phones = []
    ipa_seq = None
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            #print(f'w.lower is {w.lower()}')
            phones += lexicon[w.lower()]
        if w.lower() != ' ' and w.lower() not in lexicon:
            #print(f'w.lower() not in lexicon: {w.lower() not in lexicon}')
            ipa_list = list(ipa.convert(w))
            #print(ipa_list)
            ipa_seq = ''
            for s in ipa_list:
                #print('s is'), s == "'")
                if s != "'":
                    ipa_seq += f'{s} '
            print(f'IPA sequence: {ipa_seq}')
            #phones += list(filter(lambda p: p != " ", g2p(w)))
            phones += ipa_seq.split()
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    
    if ipa_seq != None:
        txt_seq_ipa = text_to_sequence_ipa(
                phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        #print(f'Text to sequence with IPA: {txt_seq_ipa}')
        sequence = np.array(
            text_to_sequence_ipa(
                phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )
    else:
        sequence = np.array(
            text_to_sequence(
                phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )

    return np.array(sequence)

def map_age_to_idx(age):
    with open(f'{preprocess_config["path"]["preprocessed_path"]}/ages.json') as f:
            ages_id_map = json.load(f)
    age = ages_id_map[age]
    return age

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():    
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="speaker ID for multi-speaker synthesis",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--age_control",
        type=str,
        default=None,
        help="control the age of the speaker",
    )
    args = parser.parse_args()
        
    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    
    # Check age control argument
    with open(f'{preprocess_config["path"]["preprocessed_path"]}/ages.json') as f:
            ages_id_map = json.load(f)
    if args.age_control is not None:
        valid_age_groups = list(ages_id_map.keys())
        if args.age_control.lower() not in valid_age_groups:
            print(f"Error: Invalid age group. Please choose from: {valid_age_groups}.")
            exit(1)
        else:
            print(f"Perceived syntehsised age: {args.age_control}")
    
    # Check speaker id validity
    with open(f'{preprocess_config["path"]["preprocessed_path"]}/speakers.json') as f:
        speaker_id_map = json.load(f)

    if args.speaker_id in speaker_id_map:
        speaker_id = speaker_id_map[args.speaker_id]
    else:
        print(f"Error: Invalid speaker ID '{args.speaker_id}'.")
        #exit(1)
    
    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        if args.age_control is None:
            age = np.array([map_age_to_idx('adult')])
        if args.age_control is not None:
            age = np.array([map_age_to_idx(args.age_control)])
        batchs = [(ids, raw_texts, speakers, age, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)