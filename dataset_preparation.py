''' The following code aims to generate all the necessary files for the training
of the AgingFastSpeech2 with CommonVoice datasets. Additionallt, it provides 
information about the chosen data, such as the number of utterances per age group.
'''

from utils.dataset import DatasetPreparation

dataset_name = 'CV_17'
create_folder = False
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
    #new_age_group_file = 'new_age_group_CV_17_selected_validated_sp_renamed.txt' #TEST PURPOSE ONLY, TO BE DELETED
    if duration_filelist:
        print(f'Extracting durations from {duration_filelist}...')
        total_duration = dp.get_durations_from_file(duration_filelist, new_age_group_file)
    
    else:
        print('Extracting durations from the input folder...')
        try:
           # Provisional unbalanced dataset to extract the durations
           dp.create_dataset(new_age_group_file, input_dir, output_dir_1)
           
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
      dp.create_dataset(balanced_dataset_file, input_dir, output_dir_2)

if 'myst' in dataset_name.lower():
    
    input_dir = 'myst...'
    output_dir = 'MyST_prepared'
    
    dataset_txt = dp.prepare_myst_files(input_dir)
    
    dp.create_dataset(dataset_txt, input_dir, output_dir)
    