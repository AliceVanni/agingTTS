from utils.generate_lab_files import create_labs
from utils.agingTTS_dataset import AgingTTSdataset

main_directory = 'agingTTS' # Folder with the audio files
file_mapping = 'agingTTS.txt' # Text file with the dataframe
                                      # (client_id, filename, transcription and age)

create_labs(main_directory, file_mapping)
#AgingTTSdataset().create_age_files(main_directory, file_mapping)