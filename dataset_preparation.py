''' The following code aims to generate all the necessary files for the training
of the AgingFastSpeech2 with CommonVoice datasets. Additionallt, it provides 
information about the chosen data, such as the number of utterances per age group.
'''

from utils.dataset import DatasetPreparation

dataset_name = 'MyST'
dp = DatasetPreparation(dataset_name)
    
if 'cv' in dataset_name.lower():

    all_utterances_file = 'C:/Users/Alice/validated.tsv'
    input_dir = 'C:/Users/Alice/cv-corpus-17.0-delta-2024-03-15-en.tar/cv-corpus-17.0-delta-2024-03-15/en/clips'
    output_dir_1 = 'CV_17_prepared_unbalanced'
    output_dir_2 = 'CV_17_prepared_balanced'
    
    selected_utterances = dp.select_utterances_from_cv(all_utterances_file)
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
    
    # Provisional unbalanced dataset to extract the durations
    dp.create_dataset(new_age_group_file, input_dir, output_dir_1)
    
    # Getting information about the duration of each file
    full_duration_unbalanced = dp.get_audio_folder_duration(output_dir_1, new_age_group_file)
    
    # Generating the file with the dataset with utterances balanced on gender
    # and age groups
    balanced_dataset_file = dp.select_balanced_utterances(new_age_group_file, dataset_name)
    
    # Printing the number of speakers and utterances per age group of the balanecd dataset
    age_utterances_df, age_speakers_df = dp.count_speaker_per_group(balanced_dataset_file, 'age')
    print(f'Utterances per age group\n{age_utterances_df}')
    print(f'\nSpeakers per age group\n{age_speakers_df}')
    
    # Printing the number of speakers and utterances per gender of the balanecd dataset
    gender_utterances_df, gender_speakers_df = dp.count_speaker_per_group(balanced_dataset_file, 'gender')
    print(f'Utterances per gender\n{gender_utterances_df}')
    print(f'\nSpeakers per gender\n{gender_speakers_df}')
    
    # Creation of the list of the audio files only (useful for extracting from tar/zip)
    dp.create_file_list('new_age_group_selected_validated_sp_renamed.txt', dataset_name)
    
    # Creation of the directory with the selected files
    dp.create_dataset(balanced_dataset_file, input_dir, output_dir_2)
    
    # Getting information about the duration of each file
    full_duration = dp.get_audio_folder_duration(output_dir_2, balanced_dataset_file)

if 'myst' in dataset_name.lower():
    
    input_dir = 'myst...'
    output_dir = 'MyST_prepared'
    
    dataset_txt = dp.prepare_myst_files(input_dir)
    
    dp.create_dataset(dataset_txt, input_dir, output_dir)
    