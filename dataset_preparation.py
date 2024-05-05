''' The following code aims to generate all the necessary files for the training
of the AgingFastSpeech2 with CommonVoice datasets. Additionallt, it provides 
information about the chosen data, such as the number of utterances per age group.
'''
import pandas as pd
from utils.dataset import DatasetPreparation

dataset_name = 'myST'
create_folder = False
directory_cleaning = False
dp = DatasetPreparation(dataset_name)
    
if 'cv' in dataset_name.lower():

    all_utterances_file = 'cv-corpus-17.0-2024-03-15/en/validated.tsv'
    duration_filelist = 'cv-corpus-17.0-2024-03-15/en/clip_durations.tsv'
    input_dir = 'cv-corpus-17.0-delta-2024-03-15/en/clips'
    output_dir_1 = 'CV_17_prepared_unbalanced'
    output_dir_2 = 'CV_17_prepared_balanced'
    
    selected_utterances = dp.select_utterances_from_cv(all_utterances_file, dataset_name)
    renamed_speakers_file = dp.rename_speaker_id(selected_utterances)       
    new_age_group_file = dp.age_group_redefinition(renamed_speakers_file)
    
    # Printing the number of speakers and utterances per age group
    age_utterances_df, age_speakers_df = dp.count_speaker_per_group(new_age_group_file, 'age')
    print(f'Utterances per age group\n{age_utterances_df}')
    print(f'\nSpeakers per age group\n{age_speakers_df}')
    
    # Printing the number of speakers and utterances per gender
    gender_utterances_df, gender_speakers_df = dp.count_speaker_per_group(new_age_group_file, 'gender')
    print(f'Utterances per gender\n{gender_utterances_df}')
    print(f'\nSpeakers per gender\n{gender_speakers_df}')
    
    # Getting the duration of the audio clips from the provided txt list, if present
    if duration_filelist:
        print(f'Extracting durations from {duration_filelist}...')
        total_duration = dp.get_durations_from_file(duration_filelist, new_age_group_file)
    
    else:
        print('Extracting durations from the input folder...')
        try:
           # Provisional unbalanced dataset to extract the durations
           dp.create_dataset_from_cv(new_age_group_file, input_dir, output_dir_1)
           
           # Getting information about the duration of each file
           full_duration_unbalanced = dp.get_audio_folder_duration(output_dir_1, new_age_group_file)
           
           # Generating the file with the dataset with utterances balanced on gender
           # and age groups
           balanced_dataset_file = dp.select_balanced_utterances(new_age_group_file, dataset_name)
           
        except:
             print('No dataset input folder provided, impossible to extract the durations from the data')
             raise
             
    # Generating the file with the dataset with utterances balanced on gender
    # and age groups based on the duration of the audio clips
    balanced_dataset_file = dp.select_balanced_utterances(new_age_group_file, dataset_name)
  
    # Printing the number of speakers and utterances per age group of the balanecd dataset
    age_utterances_df, age_speakers_df = dp.count_speaker_per_group(balanced_dataset_file, 'age')
    print(f'Utterances per age group in the balanced dataset\n{age_utterances_df}')
    print(f'\nSpeakers per age group in the balanced dataset\n{age_speakers_df}')
    
    # Printing the number of speakers and utterances per gender of the balanecd dataset
    gender_utterances_df, gender_speakers_df = dp.count_speaker_per_group(balanced_dataset_file, 'gender')
    print(f'Utterances per gender\n{gender_utterances_df}')
    print(f'\nSpeakers per gender\n{gender_speakers_df}')
    
    # Creation of the list of the audio files only (useful for extracting from tar/zip)
    dp.create_file_list(balanced_dataset_file, dataset_name)
      
    if create_folder == True:
      # Creation of the directory with the selected files
      dp.create_dataset_from_cv(balanced_dataset_file, input_dir, output_dir_2)
      
    if directory_cleaning == True:
      missing_filename, deleted_filename = dp.corpus_directory_cleaning(balanced_dataset_file, output_dir_2, corpus_name=dataset_name)

if 'myst' in dataset_name.lower():
    
    input_dir = 'myst_child_conv_speech_LDC2021S05\\myst_child_conv_speech\\myst_child_conv_speech_data' # TO BE ADJUSTED
    final_dir = 'MyST_prepared'
    txt_files_output_dir = 'myst_child_conv_speech_speakers_data'
    
    dp.prepare_myst_files(input_dir, txt_files_output_dir)
    myst_transcribed_file_txt = dp.myst_list_transcribed(txt_files_output_dir)
    myst_clean_file = dp.myst_cleaning(myst_transcribed_file_txt)
    
    n_speakers = dp.count_speaker_id(myst_clean_file)
    n_utterances_per_speaker_df, _ = dp.count_speaker_per_group(myst_clean_file, 'client_id')
    
    # Saving info about the number of utterances per speaker in a txt file
    n_utterances_per_speaker_df.to_csv('myst_utterance_count.txt', sep='\t', index=True, header=True)
    print(f'Utterances per speaker id:\n{n_utterances_per_speaker_df}')
    
    if create_folder == True:    
      dp.create_dataset_from_myst(myst_clean_file, input_dir, final_dir)
      
    if directory_cleaning == True:
        missing_filename, deleted_filename = dp.corpus_directory_cleaning(myst_clean_file, final_dir, corpus_name=dataset_name)
