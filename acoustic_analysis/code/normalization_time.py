import pandas as pd

filepath = 'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitchresults_bn.txt'
new_filepath = f'{filepath[:-4]}_normalized.txt'

df = pd.read_csv(filepath, delimiter='\t', header=0)

print('Original dataframe')
print(df)

def extract_speaker_id(filename):
    return filename.split('_')[0]

def normalize_time(df):
    max_time = df['Time_stamp'].max()
    df['Normalized_time'] = df['Time_stamp'] / max_time
    return df

# Add a new column for speaker ID
df['Speaker_ID'] = df['Filename'].apply(extract_speaker_id)

# Group by the new Speaker_ID column and normalize time within each group
normalized_df = df.groupby('Speaker_ID').apply(normalize_time).reset_index(drop=True)

print('Normalized dataframe')
print(normalized_df)

# Save the normalized DataFrame to a new file
normalized_df.to_csv(new_filepath, sep='\t', index=False)
