''' The following code aims to generate all the necessary files for the training
of the AgingFastSpeech2 with CommonVoice datasets. Additionallt, it provides 
information about the chosen data, such as the number of utterances per age group.
'''

from utils.dataset import select_utterances_from_cv, rename_speaker_id, create_dataset, count_speaker_per_group, age_group_redefinition, select_balanced_utterances, create_file_list, get_audio_folder_duration

all_utterances_file = 'validated_cv_en_17.tsv'
input_dir = 'cv-corpus-17.0-delta-2024-03-15/en/clips'
output_dir = 'CV_17_Delta_prepared'
dataset_name = 'CV_'

selected_utterances = select_utterances_from_cv(all_utterances_file)
renamed_speakers_file = rename_speaker_id(selected_utterances)       
new_age_group_file = age_group_redefinition(renamed_speakers_file)

# Printing the number of speakers and utterances per age group
age_utterances_df, age_speakers_df = count_speaker_per_group(new_age_group_file, 'age')
print(f'Utterances per age group\n{age_utterances_df}')
print(f'\nSpeakers per age group\n{age_speakers_df}')

# Printing the number of speakers and utterances per gender
gender_utterances_df, gender_speakers_df = count_speaker_per_group(new_age_group_file, 'gender')
print(f'Utterances per gender\n{gender_utterances_df}')
print(f'\nSpeakers per gender\n{gender_speakers_df}')

# Generating the file with the dataset with utterances balanced on gender
# and age groups
balanced_dataset_file = select_balanced_utterances(new_age_group_file, dataset_name)

# Printing the number of speakers and utterances per age group of the balanecd dataset
age_utterances_df, age_speakers_df = count_speaker_per_group(balanced_dataset_file, 'age')
print(f'Utterances per age group\n{age_utterances_df}')
print(f'\nSpeakers per age group\n{age_speakers_df}')

# Printing the number of speakers and utterances per gender of the balanecd dataset
gender_utterances_df, gender_speakers_df = count_speaker_per_group(balanced_dataset_file, 'gender')
print(f'Utterances per gender\n{gender_utterances_df}')
print(f'\nSpeakers per gender\n{gender_speakers_df}')

# Creation of the list of the audio files only (useful for extracting from tar/zip)
create_file_list(balanced_dataset_file, dataset_name)

# Creation of the directory with the selected files
create_dataset(balanced_dataset_file, input_dir, output_dir)

# Getting information about the duration of each file
full_duration = get_audio_folder_duration(output_dir, balanced_dataset_file)
