'''Creation of the final dataset on which AgingTTS is trained'''

import os
import librosa

from utils.agingTTS_dataset import AgingTTSdataset
from utils.generate_lab_files import create_labs

attsdp = AgingTTSdataset()
#directory_list = ['FilteredCV17', 'FilteredMyST']
directory_list = ['FilteredMyST']
dataframe_list = ['filtered_CV17.txt', 'filtered_myst.txt']

'''for directory in directory_list:  
    
    # Choose a random file from the list and printing its sampling rate
    first_folder = os.listdir(directory)[0]
    file_list = os.listdir(os.path.join(directory, first_folder))
    file = file_list[2] if file_list[1].split('.')[-1] == 'lab' else file_list[1]
    random_file = os.path.join(directory, first_folder, file)
    _, sampling_rate = librosa.load(random_file, sr=None)
    print(f'The sample rate for {random_file} is {sampling_rate}')'''

# Creation of the directory
#agingtts_dataframe = attsdp.create_agingtts_dataframe(dataframe_list)
agingtts_directory = attsdp.create_agingtts_dataset(directory_list)

# Changing the audio format and resampling at 16 kHz
attsdp.audio_format_to_wav(agingtts_directory)

'''# Create labs and age files
create_labs(agingtts_directory, agingtts_dataframe)
attsdp.create_age_files(agingtts_directory, agingtts_dataframe)'''