'''CREATION OF THE DATASET FOR AGINGTTS
In this file I defined useful functions to generated the directory to train
the agingTTS system I developed based on FastSpeech2.

It adjusts file formats, sample rate, dataframe files and creates the final 
directory.'''

import soundfile as sf

import os
import shutil
import pandas as pd

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
    
    def is_tab_separated(self, filename):
        '''Checks if the input file is a tab-separated file.
        
        Returns bool'''
        
        with open(filename, 'r') as f:
            line = f.readline()
            return '\t' in line
    
    def create_agingtts_dataframe(self, corpus_txt_list, new_filename = 'agingTTS.txt'):
        
        '''Creates the file with the information of all the data used for
        the training of the TTS system.
        
        Takes in input a list tab-separated txt files with the information 
        about the datasets.
        
        Input:
            - list of filenames (type=list)
            - name of the new file (type=str), default is agingTTS
        Output: a text file with the full list of files.
        
        Returns the name of the output file.
        '''
        
        df_columns = ['client_id', 'path',
                      'sentence', 'age', 'gender', 'accents']
        new_df = pd.DataFrame(columns=df_columns)
        
        for txt in corpus_txt_list:
            if not os.path.exists(txt):
                print(f"Error: {txt} does not exist")
                return None
            if not txt.endswith('.txt') or not self.is_tab_separated(txt):
                print(f"Error: {txt} is not a tab-separated txt file")
                return None
            
            txt_df = pd.read_csv(txt, sep='\t', header=False)
            new_df = pd.concat([new_df, txt_df], ignore_index=True)

        new_df.to_csv(new_filename, sep='\t', index=False, header=True)
                
        print(f'New datafarame correctly generated and saved as {new_filename}')
        
        return new_filename
        
    def create_agingtts_dataset(self, corpus_directory_list, new_directory_name='agingTTS'):
        
        '''Creates the directory with all the necessary files for agingTTS
        training by merging the list of directories in input.
        
        Input: 
            - list of directories (type=list)
            - name of the new directory (type=str), default is agingTTS
        
        Returns None
        '''
        num_folders = 0
        for corpus_directory in corpus_directory_list:
            
            if os.isdir(corpus_directory) == True:
                print(f'Copying files from {corpus_directory}...')
                for folder in tqdm(os.listdir(corpus_directory)):
                    num_folders += 1
                    shutil.copy2(os.path.join(corpus_directory, folder), new_directory_name)
            else:
                print(f'{corpus_directory} is not a directory')
        
        print(f'{new_directory_name} correctly created with {num_folders} folders')

            
        