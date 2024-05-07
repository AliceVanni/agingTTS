'''CREATION OF THE DATASET FOR AGINGTTS
In this file I defined useful functions to generated the directory to train
the agingTTS system I developed based on FastSpeech2.

It adjusts file formats, sample rate, dataframe files and creates the final 
directory.'''

import soundfile as sf

import os
from tqdm import tqdm

class AgingTTSdataset:
    
    def __init__(self):
        self
    
    def audio_format_to_wav(self, directory_path):
        '''Change the audio file format to .wav.
        The audio formats to be changed are mp3 and flac.
        All the audios are resampled at 1600 Hz.
        
        The fucntion assumes the folder to have only one level of subfolders.
        
        Returns None'''
        
        print('Converting the audio files...')
        # Loop over each subdirectory of the main directory
        for sub_folder in tqdm(os.listdir(directory_path)):
            
            for file in os.listdir(os.path.join(directory_path, sub_folder)):
                file_path = os.path.join(directory_path, sub_folder, file)
                
                if file.endswith(('.mp3', '.flac')):
                    source_audio_format = file.split('.')[-1]
                    data, _ = sf.read(file_path)
                    sf.write(file_path.replace(f'.{source_audio_format}', '.wav'), data, samplerate = 1600, format = 'WAV')
        
        print(f'Convertion to .wav file of {directory_path} directory completed')

    def agingtts_dataset_creation(self, txt_corpus_1, txt_corpus_2, directory_corpus_1, directory_corpus_2):
        
        '''Creation of the file with the information of all the data used for
        the training of the TTS system.
        
        Takes in input two txt files with the information about two datasets,
        and the path to the corresponding directories.
        
        Outputs a text file with the full list of files.
        
        Returns the name of the output file.
        '''
        
        # Merge the two files in a new file called: 
        
        # Merge the two folders in a new folder called: