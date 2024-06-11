# Barpolot for GT speaking rate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_speaking_rate(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 6))
    bar_width = 0.8
    bar_positions = np.arange(len(df))
    bar_labels = df['name'].tolist()

    for idx, row in df.iterrows():
        color = plt.cm.viridis(idx / len(df))
        ax.bar(bar_positions[idx], row['speechrate(nsyll/dur)'], color=color, width=bar_width)

    # Create custom legend with color and speaker category correspondence
    unique_names = df['name'].unique()
    legend_handles = [plt.Rectangle((0,0),1,1, color=plt.cm.viridis(idx / len(df))) for idx, word in enumerate(unique_names)]
    #ax.legend(legend_handles, unique_names, title='Files', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Set xticks and labels
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax.set_xlabel('Name')
    ax.set_ylabel('Speaking Rate (syllables/second)')
    ax.set_title('Speaking Rate for Ground Truth')

    plt.tight_layout()
    plt.show()

# Path to your CSV file
csv_file = '../speaking_rate/SyllableNuclei_gt.txt'

# Call the function to plot speaking rates
plot_speaking_rate(csv_file)
