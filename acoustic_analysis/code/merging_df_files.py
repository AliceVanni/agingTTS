#merging back the normalized df

import pandas as pd


file_paths = ['D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitchresults_base_normalized.txt',
              'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitchresults_bn_normalized.txt',
              'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitchresults_gt_normalized.txt']

dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path, delimiter='\t')
    dataframes.append(df[1:])  # Skip the header row and add the rest to the list

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

merged_output_file = 'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/full_pitchresults_20_450.txt'
merged_df.to_csv(merged_output_file, sep='\t', index=False)

print(f"Merged data saved {merged_output_file}")