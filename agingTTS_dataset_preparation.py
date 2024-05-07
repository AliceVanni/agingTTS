'''Creation of the final dataset on which AgingTTS is trained'''
import os
import librosa

from utils.agingTTS_dataset import AgingTTSdataset

attsdp = AgingTTSdataset()
directory_list = ['FilteredCV17', 'FilteredMyST']
dataframe_list = ['filtered_CV17.txt', 'filtered_myst.txt']

for directory in directory_list:  
    
    # Choose a random file from the list and printing its sampling rate
    first_folder = os.listdir(directory)[0]
    file_list = os.listdir(os.path.join(directory, first_folder))
    file = file_list[1] if file_list[0].split('.')[-1] == 'lab' else file_list[0]
    random_file = os.path.join(directory, first_folder, file)
    _, sampling_rate = librosa.load(random_file, sr=None)
    print(f'The sample rate for {random_file} is {sampling_rate}')
    
    # Changing the audio format and rresampling at 16k Hz
    attsdp.audio_format_to_wav(directory)

dataframe_filename = attsdp.create_agingtts_dataframe(dataframe_list)
attsdp.create_agingtts_dataset(directory_list)