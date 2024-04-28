# Selection of CV sub-dataset based on age and gender criteria:
    # - Balance in terms of gender
    # - Removal of utterance from 'teens'
    # - Removal of utterances from 'sixties'
# And creation of the corresponding filelist (output_filelist) with new speaker
# id. The selected utterances are listed in the file selected_utterances_filelist.
# Other useful functions to quickly obtain information about the dataset.

from tqdm import tqdm
import pandas as pd
import csv
import os
import shutil

def count_speaker_id(input_file):
    
    '''Counts the number of speakers based on a txt list (i.e. input_file) 
    of utterances in the corpus
    
    Returns the number of speaker (type: int) and a list with the all the 
    speaker ids'''
    
    with open(input_file, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    
    og_sp_id_list = []
    
    for line in tqdm(lines):
        split_line = line.split('\t')
        og_sp_id = split_line[0]
        
        if og_sp_id not in og_sp_id_list:
            og_sp_id_list.append(og_sp_id)
            
    num_speaker_id = len(og_sp_id_list)    
    print(f'There are {num_speaker_id} speakers in the provided list')
    
    return num_speaker_id, og_sp_id_list
        
def rename_speaker_id(input_file, output_file):
    
    '''Renames the speaker id with integer numbers padded left with 
    a number of zeros equal to the longest number.
    Returns a file equal to the input one, with the new speaker ids'''
    
    print('Getting the list of speakers...')
    num_speaker_id, og_sp_id_list = count_speaker_id(input_file)
    num_pads = len(str(num_speaker_id))
    
    new_sp_id_dict = {}
    new_sp_id = 0
    print('Creating the new speaker IDs...')
    for speaker_id in tqdm(og_sp_id_list):
        new_sp_id_dict[speaker_id] = str(new_sp_id).zfill(num_pads)
        new_sp_id += 1
    
    print('Writing the output file with the new speaker IDs...')
    with open(input_file, 'r', encoding='utf-8') as in_file, \
        open(output_file, 'w', encoding='utf-8', newline='') as out_file:
            
        reader = csv.reader(in_file, delimiter='\t')
        writer = csv.writer(out_file, delimiter='\t')

        writer.writerow(next(reader))
    
        for row in tqdm(reader):
            old_sp_id = row[0]
            new_sp_id = new_sp_id_dict.get(old_sp_id, old_sp_id)
            row[0] = new_sp_id
            writer.writerow(row)
            
    print(f'\nThe file {output_file} is ready')
    return 

def select_utterances_from_cv(all_utterances_file, output_file):
    '''Generates a list of utterances with the correct criteria:
        - Age != 'teens', 'sixties' and not empty
        - Gender != empty and equal balance
    Removing the columns that are not relevant. This assumes that there are
    columns calles 'client_id', 'path', 'sentence', 'age', 'gender', 'accents'
    as standard in the CommonVoice dataset files
    It returns the name of the output file.'''
    
    all_utterances_df = pd.read_csv(all_utterances_file, delimiter='\t', header=0)
    print('Overview of the raw dataset:')
    print(all_utterances_df.head(5))
    
    #Initialising the clean df (the useless columns will be deleted manually)
    clean_cols = ['client_id', 'path', 'sentence', 'age', 'gender', 'accents']
    clean_df = pd.DataFrame(columns=clean_cols)
    selected_rows = all_utterances_df[pd.notna(all_utterances_df['age']) & (all_utterances_df['age'] != 'teens') & (all_utterances_df['age'] != 'sixties') & pd.notna(all_utterances_df['gender'])]
    clean_df = clean_df.append(selected_rows[clean_cols], ignore_index=True)
    print('Overview of the clean dataset:')
    print(clean_df.head(5))
        
    clean_df.to_csv(output_file, sep='\t', index=False, header=True)
    print(f'The clean dataframe was successfully saved as {output_file}')
    
    return output_file

def create_dataset(txt_list_of_files, input_dir, output_dir):

    '''Create a new directory with the required structure for multispeaker FastSpeech2 training:
    - main directory
      - speaker 1
        - file1
        - file 2
      - speaker 2
        - file1
        - file 2
      etc.
      The creation of such a directory is based on the list of files that has to be in the directory
      provided as the first argument.
      
      Returns None.'''
      
    if not os.path.exists(output_dir):
        print('Creating the output directory...')
        os.makedirs(output_dir)
        
    file_list_df = pd.read_csv(txt_list_of_files, sep='\t', header=0)
    
    print('Seelecting the relevant files...')
    for i, row in tqdm(file_list_df.iterrows()):
        # Get the client_id and file_path from the row
        client_id = row['client_id']
        file_path = row['path']
    
        # Construct the full file path
        full_file_path = os.path.join(input_dir, file_path)
    
        # Construct the output directory path
        output_dir_path = os.path.join(output_dir, str(client_id))
    
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
    
        # Construct the output file path
        output_file_path = os.path.join(output_dir_path, file_path)
    
        # Copy the file to the output directory
        shutil.copy(full_file_path, output_file_path)
    
    print(f'The directory {output_dir} was correctly generated.')
        
def create_file_list(file):
    
    '''Create a txt file with only the list of the name of the files
    of the dataset
    
    Returns None'''
    
    with open(file, 'r', encoding='utf-8') as file_list:
        lines = file_list.readlines()
    
    list_of_files = []
    for line in lines[0:]:
        line = line.split('\t')
        list_of_files.append(line[1])
        
    with open('list_of_files.txt', 'w', encoding='utf-8') as output:
        for item in list_of_files:
            output.write("%s\n" % item)
            
def count_speaker_per_group(dataset_txt, column_name):
    '''Count the number of speakers for each group, i.e. category in the 
    column with the name given as the second argument and the number 
    of utterances for that speaker.
    
    Returns two pandas dataframe with the counts of utterances per group
    and the count of speakers per group'''
    
    dataset_df = pd.read_csv(dataset_txt, sep='\t', header=0)
    print('Overview of the dataset:')
    print(dataset_df.head(5))
    
    group_utterances_df = dataset_df[column_name].value_counts()
    group_speakers_df = dataset_df.groupby(column_name)['client_id'].nunique().reset_index(name='counts')
    
    return group_utterances_df, group_speakers_df

def age_group_redefinition(input_file, output_filename):
    
    '''Creates a new txt file, named as the second argument, in which the original
    age groups of the first argument are remapped based on the following criteria:
        - < 12: children
        - 20-50: adults
        - > 70: seniors'''
        
    original_df = pd.read_csv(input_file, sep='\t', header=0)
    new_df = original_df.copy()
    
    adult_age_groups = ['twenties', 'thirties', 'fourties', 'fifties']
    new_df.loc[new_df['age'].isin(adult_age_groups), 'age'] = 'adults'
    
    senior_age_groups = ['seventies', 'eighties', 'nineties']
    new_df.loc[new_df['age'].isin(senior_age_groups), 'age'] = 'senior'
    
    new_df.to_csv(output_filename, sep='\t', index=False, header=True)
    print(f'The redefined dataframe is saved in {output_filename}')
        