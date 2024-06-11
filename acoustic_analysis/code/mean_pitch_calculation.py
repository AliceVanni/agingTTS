# Calculate the mean, max, min and range F0 from the pitch listing

import pandas as pd

filepath = '../pitch/pitchresults_gt_normalized.txt'
new_file = '../pitch/mean_pitch_gt_model.txt'
df = pd.read_csv(filepath, sep = "\t")
print('Original dataframe')
print(df.head())

#Mean
mean_df = df.groupby(['Filename'], as_index=False)['Pitch_Hz'].mean()
mean_df.rename(columns={'Pitch_Hz': 'Mean_Pitch'}, inplace=True)

# Max
max_df = df.groupby(['Filename'], as_index=False)['Pitch_Hz'].max()
max_df.rename(columns={'Pitch_Hz': 'Max_Pitch'}, inplace=True)

# Min
min_df = df.groupby(['Filename'], as_index=False)['Pitch_Hz'].min()
min_df.rename(columns={'Pitch_Hz': 'Min_Pitch'}, inplace=True)

# Variance
variance_df = df.groupby(['Filename'], as_index=False)['Pitch_Hz'].var()
variance_df.rename(columns={'Pitch_Hz': 'Variance_Pitch'}, inplace=True)

# Concatenate dataframes
result_df = pd.concat([mean_df, max_df, min_df, variance_df], axis=1)

# Drop duplicate 'Filename' columns
result_df = result_df.loc[:,~result_df.columns.duplicated()]

#Range
def range_calculus(df):
    df['Range'] = df['Max_Pitch'] - df['Min_Pitch']

    return df

range_df = result_df.groupby('Filename').apply(range_calculus)

print('Final dataframe')
print(range_df.head())

range_df.to_csv(new_file, sep='\t', index=False)