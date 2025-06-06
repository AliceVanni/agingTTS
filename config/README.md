# Config
Here are the config files used to train the multi-speaker TTS model.
Only one configuration is given as example: agingTTS.

The following text explains some important hyper-parameters.

## preprocess.yaml
- **path.lexicon_path**: the lexicon (which maps words to phonemes) used by Montreal Forced Aligner. 
  We provide an English lexicon
- **path.df_path**: path to the tab-separated text file with the corpus dataframe, which should include age information about the speakers. The speakers id (column 'client_id') must corresponds to the speakers ids used for the folders.
- **mel.stft.mel_fmax**: set it to 8000 if HiFi-GAN vocoder is used, and set it to null if MelGAN is used.
- **pitch.feature & energy.feature**: the original paper proposed to predict and apply frame-level pitch and energy features to the inputs of the TTS decoder to control the pitch and energy of the synthesized utterances. 
  However, in our experiments, we find that using phoneme-level features makes the prosody of the synthesized utterances more natural.
- **pitch.normalization & energy.normalization**: to normalize the pitch and energy values or not. 
  The original paper did not normalize these values.

## train.yaml
- **optimizer.grad_acc_step**: the number of batches of gradient accumulation before updating the model parameters and call optimizer.zero_grad(), which is useful if you wish to train the model with a large batch size but you do not have sufficient GPU memory.
- **optimizer.anneal_steps & optimizer.anneal_rate**: the learning rate is reduced at the **anneal_steps** by the ratio specified with **anneal_rate**.

## model.yaml
- **transformer.decoder_layer**: the original paper used a 4-layer decoder, but we find it better to use a 6-layer decoder, especially for multi-speaker TTS.
- **variance_embedding.pitch_quantization**: when the pitch values are normalized as specified in ``preprocess.yaml``, it is not valid to use log-scale quantization bins as proposed in the original paper, so we use linear-scaled bins instead. 
- **multi_speaker**: to apply a speaker embedding table to enable multi-speaker TTS or not.
- **vocoder.speaker**: should be set to 'universal' if any dataset other than LJSpeech is used."

  Source: https://github.com/ming024/FastSpeech2
