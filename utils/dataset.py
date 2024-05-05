# Selection of CV sub-dataset based on age and gender criteria:
    # - Balance in terms of gender
    # - Removal of utterance from 'teens'
    # - Removal of utterances from 'sixties'
# And creation of useful files with information about the dataset.

from tqdm import tqdm
import pandas as pd
import librosa
import csv
import os
import shutil
import datetime

class DatasetPreparation:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def count_speaker_id(self, input_file):
        
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
            
    def rename_speaker_id(self, input_file):
        
        '''Renames the speaker id with integer numbers padded left with 
        a number of zeros equal to the longest number.
        Returns a file equal to the input one, with the new speaker ids'''
        
        print(f'Getting the list of speakers from file {input_file}...')
        num_speaker_id, og_sp_id_list = self.count_speaker_id(input_file)
        num_pads = len(str(num_speaker_id))
        
        new_sp_id_dict = {}
        new_sp_id = 0
        print('Creating the new speaker IDs...')
        for speaker_id in tqdm(og_sp_id_list):
            new_sp_id_dict[speaker_id] = str(new_sp_id).zfill(num_pads)
            new_sp_id += 1
        
        print('Writing the output file with the new speaker IDs...')
        output_file = input_file.split('.')[0] + '_sp_renamed.txt'
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
        return output_file
    
    def select_utterances_from_cv(self, all_utterances_file, dataset_name):
        '''Generates a list of utterances with the correct criteria:
            - Age != 'teens'and empty
            - Gender != empty
        Removing the columns that are not relevant. This assumes that there are
        columns calles 'client_id', 'path', 'sentence', 'age', 'gender', 'accents'
        as standard in the CommonVoice dataset files
        It returns the name of the output file.'''
        
        all_utterances_df = pd.read_csv(all_utterances_file, delimiter='\t', header=0)
        print(f'Overview of the raw dataset:\nNumber of entries: {all_utterances_df.shape[0]}')
        print(all_utterances_df.head(5))
        
        #Initialising the clean df (the useless columns will be deleted manually)
        clean_cols = ['client_id', 'path', 'sentence', 'age', 'gender', 'accents']
        clean_df = pd.DataFrame(columns=clean_cols)
        selected_rows = all_utterances_df[pd.notna(all_utterances_df['age']) & (all_utterances_df['age'] != 'teens') & pd.notna(all_utterances_df['gender'])]
        clean_df = clean_df.append(selected_rows[clean_cols], ignore_index=True)
        print('Overview of the clean dataset:\nNumber of entries: {all_utterances_df.shape[0]}')
        print(clean_df.head(5))
        
        all_utterances_file_name = all_utterances_file.split('/')[-1].split('.')[0]
        output_file = dataset_name + '_selected_' + all_utterances_file_name + '.txt'
        clean_df.to_csv(output_file, sep='\t', index=False, header=True)
        print(f'The clean dataframe was successfully saved as {output_file}')
        
        return output_file
    
    def age_group_redefinition(self, input_file):
        
        '''Creates a new txt file, named as the second argument, in which the original
        age groups of the first argument are remapped based on the following criteria:
            - < 12: children
            - 20-50: adults
            - > 60: seniors'''
            
        original_df = pd.read_csv(input_file, sep='\t', header=0)
        new_df = original_df.copy()
        
        adult_age_groups = ['twenties', 'thirties', 'fourties', 'fifties']
        new_df.loc[new_df['age'].isin(adult_age_groups), 'age'] = 'adult'
        
        senior_age_groups = ['sixties', 'seventies', 'eighties', 'nineties']
        new_df.loc[new_df['age'].isin(senior_age_groups), 'age'] = 'senior'
        
        output_filename = 'new_age_group_' + input_file
        new_df.to_csv(output_filename, sep='\t', index=False, header=True)
        print(f'The redefined dataframe is saved in {output_filename}')
        
        return output_filename
        
    def select_balanced_utterances(self, dataset_txt, dataset_name):
        '''Select a balanced number of utterances from male and female speakers for 
        each age group.
        The function outputs a txt file with the selected files and all the columns
        of the input file.
        The output file will have the name of the dataset, input as the second argument.
        '''
        dataset_df = pd.read_csv(dataset_txt, sep='\t', header=0)
    
        # Count the number of utterances per age group and gender
        utterances_per_age_gender = dataset_df.groupby(['age', 'gender']).size().reset_index(name='utterances')
        print('Number of utterances per age and gender in the input dataset:')
        print(utterances_per_age_gender.head())
        
        # Calculate the total duration per age group and gender
        total_duration_per_age_gender = dataset_df.groupby(['age', 'gender'])['duration'].sum().reset_index(name='duration')
        print('\nTotal duration of data per age and gender in the input dataset:')
        print(total_duration_per_age_gender.head())
        
        # Find the minimum number of utterances per age group and gender
        min_duration = total_duration_per_age_gender['duration'].min()
        print(f'\nDuration of the new groups: {min_duration}')
    
        # Select a balanced number of utterances from male and female speakers for 
        # each age group
        # Initialising the df
        selected_files_df = pd.DataFrame(columns=dataset_df.columns)
    
        for (age, gender), group in dataset_df.groupby(['age', 'gender']):
            
            # # Select the first `min_utterances` rows from the balanced group
            # group_selected = group.head(min_utterances)
    
            # selected_files_df = pd.concat([selected_files_df, group_selected], ignore_index=True)
              
            # Calculate the cumulative duration for each group
            group['cumulative_duration'] = group['duration'].cumsum()
    
            # Select the rows until the cumulative duration reaches the minimum total duration
            group_selected = group[group['cumulative_duration'] <= min_duration]
            
            group_selected = group_selected.drop('cumulative_duration', axis=1)
            selected_files_df = pd.concat([selected_files_df, group_selected], ignore_index=True)
        
        print('Data duration per age and gender in the balanced dataset:')
        print(selected_files_df.groupby(['age', 'gender'])['duration'].sum().reset_index(name='duration').head())
        
        # Save the selected files to a txt file    
        output_filename = 'duration_balanced_' + dataset_name + '.txt'
        selected_files_df.to_csv(output_filename, sep='\t', index=False, header=True)
        
        print(f'Balanced dataset txt file generated and saved as {output_filename}')
        return output_filename
    
    def create_dataset_from_cv(self, txt_list_of_files, input_dir, output_dir):
    
        '''Create a new directory with the required structure for multispeaker 
        FastSpeech2 training from the CV corpus:
        - main directory
          - speaker 1
            - file1
            - file 2
          - speaker 2
            - file1
            - file 2
          etc.
          The creation of such a directory is based on the list of files that has
          to be in the directory provided as the first argument.
          
          Returns None.'''
          
        if not os.path.exists(output_dir):
            print('Creating the output directory...')
            os.makedirs(output_dir)
            
        file_list_df = pd.read_csv(txt_list_of_files, sep='\t', header=0)
        
        print('Selecting the relevant files...')
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
                
    def count_speaker_per_group(self, dataset_txt, column_name):
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
    
    def create_file_list(self, file, dataset_name, prefix='clips'):
        
        '''Create a txt file with only the list of the name of the files
        of the given dataset. The name is specified in the second argument.
        
        Returns None'''
        
        with open(file, 'r', encoding='utf-8') as file_list:
            lines = file_list.readlines()
        
        list_of_files = []
        for line in lines[1:]:
            line = line.split('\t')
            list_of_files.append(line[1])
        
        file_name = dataset_name + '_list_of_files.txt'
        with open(file_name, 'w', encoding='utf-8') as output:
            for item in list_of_files:
                output.write(f'{prefix}/{item}\n')
        
        print(f'Generated list of files, saved as {file_name} with the prefix {prefix}')
        return list_of_files
    
    def get_audio_duration(self, file):
        
        '''
        Returns the duration in seconds of the audio file given as input.
        '''
        
        audio_data, sample_rate = librosa.load(file, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        return duration
    
    def is_audio_file(self, filename):
        
        '''Check if the file is an audio file based on its extension.'''
        
        valid_extensions = ['.mp3', '.wav', '.flac']
        _, ext = os.path.splitext(filename)
        return ext.lower() in valid_extensions
            
    def get_audio_folder_duration(self, audio_directory, dataset_info):
        '''Takes in input a directory, that might have subdirectories, and a txt/csv 
        file with the datatset information, including the list of files, and add the
        duration information about the listed files in the dataset txt.
        
        Returns the full duration of the dataset'''
        
        print(f'Getting information from {audio_directory} folder')
        
        dataset_name = dataset_info.split('.')[0]
        list_of_files = self.create_file_list(dataset_info, dataset_name)
            
        audio_duration_dict = {}
        full_duration = 0
        
        for e in os.listdir(audio_directory):
    
            if os.path.isdir(os.path.join(audio_directory, e)) == True:
                for file in os.listdir(os.path.join(audio_directory, e)):
                    
                    if file in list_of_files:
                        if self.is_audio_file(os.path.join(audio_directory, e, file)) == True:
                            
                            duration = self.get_audio_duration(os.path.join(audio_directory, e, file))
                            audio_duration_dict[file] = duration                    
                            full_duration += duration
                        
            if os.path.isfile(os.path.join(audio_directory, e)) == True: 
                if self.is_audio_file(os.path.join(audio_directory, e)) == True:
                    
                    duration = self.get_audio_duration(os.path.join(audio_directory, e))
                    audio_duration_dict[e] = duration
                    full_duration += duration
                    
        # Add the duration information to the corresponding row in the dataset_info file
        dataset_df = pd.read_csv(dataset_info, sep='\t')
        dataset_df['duration'] = dataset_df['path'].map(audio_duration_dict)  # convert column to integer type
        dataset_df['duration'] = dataset_df['duration'].astype(float)
        dataset_df.to_csv(dataset_info, sep='\t', index=False, header=True)
    
        full_duration_hms = str(datetime.timedelta(seconds=full_duration))
        print(f'Information fully retrieved and added to {dataset_info}')
        print(f'The full duration of the files in the directory is {full_duration} seconds, i.e. {full_duration_hms}')
       
        return full_duration_hms
        
    def get_durations_from_file(self, duration_file, dataset_info):
        
        '''Takes as input a the tsv file with the duration of the clips and
        adds them to the file with the metadata of the dataset given as the 
        second argument. 
        
        Return the full duration of the dataset'''
        
        # Assumes the df have two columns: one with the file name and another with
        # the duration in milliseconds
        duration_df = pd.read_csv(duration_file, sep='\t')
        print('Overview of the dataframe:')
        print(duration_df.head(10))
        
        # Renaming the columns
        print('Renaming the columns of the df...')
        duration_df = duration_df.rename(columns={duration_df.iloc[:, 0].name : 'path', duration_df.iloc[:, 1].name : 'duration'})
        print(duration_df.head(10))
        
        # Convertion from milliseconds to seconds
        duration_df['duration'] = duration_df['duration'] / 1000
        
        # Add the duration information to the corresponding row in the dataset_info file
        dataset_df = pd.read_csv(dataset_info, sep='\t')
        dataset_df = dataset_df.merge(duration_df, on='path', how='left')
        print(dataset_df.head(10))
        dataset_df['duration'] = dataset_df['duration'].astype(float)
        dataset_df.to_csv(dataset_info, sep='\t', index=False, header=True)
        
        # Calculating the duration of the whole dataset
        full_duration = dataset_df['duration'].sum()
    
        full_duration_hms = str(datetime.timedelta(seconds=full_duration))
        print(f'Information fully retrieved and added to {dataset_info}\nDataset total duration: {full_duration_hms}')
        
        return full_duration_hms
        
    def prepare_myst_files(self, main_directory_path, txt_files_output_dir):
        
        '''Prepares a txt file with the information about each speaker in the
        MyST corpus. The files have the same structure as files from CV in 
        order to use them together.
        
        Takes as argument the path of the directory in which the MyST files are
        and the name of the directory in whcih to store the txt files, one for 
        each speaker.
        
        Returns None'''
                                             
        main_directory = os.listdir(main_directory_path)
        
        cv_columns = ['client_id', 'path', 'sentence', 'age', 'gender', 'accents']
        myst_df_clean = pd.DataFrame(columns=cv_columns)
        
        for partition_folder in main_directory:  
            print(f'Extracting information about partition folder {partition_folder}')
            student_id_folders = os.listdir(os.path.join(main_directory_path, partition_folder))
            
            for student in student_id_folders:
                student_df = myst_df_clean.copy(deep=True)
                print(f'Writing info about client id {student}')
                student_df['client_id'] = student
                
                for session_folder in os.listdir(os.path.join(main_directory_path, partition_folder, student)):
                    print(f'Session: {session_folder}')
                    
                    for file in os.listdir(os.path.join(main_directory_path, partition_folder, student, session_folder)):
                        print(f'File: {file}')
                        
                        if file.split('.')[-1] == 'txt':
                            continue
                        
                        if self.is_audio_file(os.path.join(main_directory_path, partition_folder, student, session_folder, file)) == True:
                            new_file_row = pd.DataFrame({'client_id' : [str(student)], 'path' : [file], 'sentence' : [''], 'age' : ['children'], 'gender' : ['not_given'], 'accents' : ['not_given']})
                            student_df = pd.concat([student_df, new_file_row], ignore_index=True)
                        
                        transcription_file = ''
                        
                        if file.split('.')[-1] == 'trn':
                            
                            transcription = open(os.path.join(main_directory_path, partition_folder, student, session_folder, file), 'r', encoding='utf-8').read().replace('\n', '')
                            
                            transcription_file = file.replace('.trn', '.lab') # Changing the extension of the transcription files
                            with open(os.path.join(main_directory_path, partition_folder, student, session_folder, transcription_file), 'w', encoding='utf-8') as txt_file:
                                txt_file.write(transcription)
                                
                            student_df.loc[student_df['path'] == file.replace('.trn', '.flac'), 'sentence'] = transcription
                            
                if not os.path.exists(txt_files_output_dir):
                    os.makedirs(txt_files_output_dir)
                    
                output_file = f'{txt_files_output_dir}/myST_corpus_{student}.txt'
                student_df.to_csv(output_file, sep='\t', index=False, header=True)
    
    def myst_list_transcribed(self, myst_speakers_files_folder):
        
        '''From the text files with the information about the MyST corpus speakers,
        add to a new txt file only the files with a corresponding transcription,
        or which transcription is just <SILENCE>.
        
        Returns the name of the output file'''
        cols = ['client_id', 'path', 'sentence', 'age', 'gender', 'accents']
        transcribed_df = pd.DataFrame(columns=cols)
        
        for file in tqdm(os.listdir(myst_speakers_files_folder)):        
            speaker_df = pd.read_csv(os.path.join(myst_speakers_files_folder, file), sep='\t')
            selected_rows = speaker_df[pd.notna(speaker_df['sentence']) & (speaker_df['sentence'] != '<SILENCE>')]
            transcribed_df = pd.concat([transcribed_df, selected_rows], ignore_index=True)
        
        output_filename = 'myst_transcribed.txt'
        transcribed_df.to_csv(output_filename, sep='\t', index=False, header=True)
        return output_filename 
    
    def myst_cleaning(self, myst_transcribed_file_txt):
        
        '''From the list of the audio files with a transcription available 
        given in input, selects only clean and relevant audio files. 
        The file is excluded if the transcription:
            - contains <INAUDIBLE>
            - contains <NO_SIGNAL> or <NO_SIGNAL
            - contains <INDISCERNIBLE>
            - is (())
            - contains or is <DISCARD>
            - contains or is <NOISE>
            
        Return the name of the output file: myst_clean.txt'''
        
        # List of critical words
        blacklist = ['<INAUDIBLE>', '<NO_SIGNAL>', '<INDISCERNIBLE>', '(())', '<DISCARD>', '<NOISE>', '<NO_SIGNAL']
        output_filename = 'myst_clean_2.txt'
        cleaned_list_str = ''
        with open(myst_transcribed_file_txt) as f:
            speaker_files_list = f.readlines()
            
        for speaker in tqdm(speaker_files_list):
            transcription = speaker.split('\t')[2]
            
            if any(word in transcription.split() for word in blacklist):
                continue
            
            else:
                # Add the whole line to a new txt file
                cleaned_list_str += f'{speaker}'
            
        with open(output_filename, 'w', encoding='utf-8') as f_clean:
            f_clean.write(cleaned_list_str)
        
        print(f'\nClean list of files created and saved as {output_filename}')
        return output_filename

    def create_dataset_from_myst(self, txt_list_of_files, input_dir, output_dir):
    
        '''Create a new directory with the required structure for multispeaker 
        FastSpeech2 training from the MyST corpus:
        - main directory
          - speaker 1
            - file1
            - file 2
          - speaker 2
            - file1
            - file 2
          etc.
          The creation of such a directory is based on the list of files that has
          to be in the directory provided as the first argument.
          
          Returns None.'''
          
        if not os.path.exists(output_dir):
            print('Creating the output directory...')
            os.makedirs(output_dir)
            
        file_list_df = pd.read_csv(txt_list_of_files, sep='\t', header=0)
        
        print(f'Main directory: {input_dir}')
        
        for i, row in tqdm(file_list_df.iterrows()):
            
            # Get the client_id and file_basename from the row
            client_id = row['client_id']
            file_basename = row['path']  # Already just the basename
            
            # Search for the file in the input directory and its subdirectories
            found = False
            for root, dirs, files in os.walk(input_dir):
                if file_basename in files:
                    found = True
                    full_file_path = os.path.join(root, file_basename)
                    break
            
            if not found:
                print(f"File '{file_basename}' not found in the input directory.")
                continue
            
            # Construct the output directory path
            output_client_dir = os.path.join(output_dir, str(client_id))
            
            # Create the output client directory if it doesn't exist
            if not os.path.exists(output_client_dir):
                os.makedirs(output_client_dir)
            
            # Copy the file to the output directory
            output_file_path = os.path.join(output_client_dir, file_basename)
            shutil.copy(full_file_path, output_file_path)
        
        print(f'The directory {output_dir} was correctly generated.')
          
        # if not os.path.exists(output_dir):
        #     print('Creating the output directory...')
        #     os.makedirs(output_dir)
            
        # file_list_df = pd.read_csv(txt_list_of_files, sep='\t', header=0)
        # file_list = []
        
        # print(f'Main directory: {input_dir}')
        
        # # Getting the full list of the file paths to be copyed
        # for partition in os.listdir(input_dir):
            
        #     for speaker in os.listdir(os.path.join(input_dir, partition)):
        #         print(f'Speaker: {speaker}')
                
        #         for session in os.listdir(os.path.join(input_dir, partition, speaker)):
        #             print(f'Session: {session}')
                    
        #             for file in os.listdir(os.path.join(input_dir, partition, speaker, session)):
                       
        #                 if file.split('.')[-1] == 'flac' or file.split('.')[-1] == 'lab':
        #                     print(f'File: {file}')
        #                     full_file_path = os.path.join(input_dir, partition, speaker, session, file)
        #                     file_list.append(full_file_path)
                            
        # print('Selecting the relevant files...')
        
        # for i, row in tqdm(file_list_df.iterrows()):
            
        #     # Get the client_id and file_path from the row
        #     client_id = row['client_id']
        #     file_path = row['path']
        
        #     # Construct the output directory path
        #     output_dir_path = os.path.join(output_dir, str(client_id))
        
        #     # Create the output directory if it doesn't exist
        #     if not os.path.exists(output_dir_path):
        #         os.makedirs(output_dir_path)
                
        #     # Construct the output file path
        #     output_file_path = os.path.join(output_dir_path, file_path)
        
        #     # Copy the file to the output directory
        #     shutil.copy(full_file_path, output_file_path)
        
        # print(f'The directory {output_dir} was correctly generated.')