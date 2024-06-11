# Pitch normalization per speaker

import pandas as pd

filepath = 'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitchresults_gt_normalized.txt'

df = pd.read_csv(filepath, '\t')

df['Pitch_Hz'] = pd.to_numeric(df['Pitch_Hz'], errors='coerce')

# Function to normalize 'Pitch_Hz' within each recording
def normalize_pitch(df):
    df['Normalized_pitch'] = df['Pitch_Hz'] / df['Pitch_Hz'].mean()

    return df

# Applying normalization based on 'Speaker_ID'
normalized_df = df.groupby('Speaker_ID').apply(normalize_pitch)

print(normalized_df)

normalized_df.to_csv(filepath, sep='\t', index=False)