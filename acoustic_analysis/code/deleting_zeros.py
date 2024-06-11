import pandas as pd

file_path = 'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitchresults_base_normalized.txt'

# # Read the file
# with open(file_path, 'r') as file:
#     lines = file.readlines()
# #print(lines)

# # Filter out rows with '0.0' in 'Pitch (Hz)' column
# for line in lines:
#     line = line.split('\t')
    
# modified_lines = [line for line in lines if line[2] != '0.0']

# # Write the modified lines back to the file
# with open(file_path, 'w') as file:
#     file.writelines(modified_lines)

df = pd.read_csv(file_path, delimiter='\t', header=0)

filtered_df = df[df['Pitch_Hz'] != '--undefined--']

print(f'Filtered dataframe:\n{filtered_df}')

filtered_df.to_csv(file_path, sep='\t', index=False)