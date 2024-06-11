import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_speaking_rate(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    model_name = csv_file.split("/")[-1].split(".")[0].split("_")[-1]
    
    # Extract the first part of 'name' as the grouping key
    df['group'] = df['name'].apply(lambda x: x.rsplit('_', 2)[0])

    # Get unique groups
    groups = df['group'].unique()
    num_groups = len(groups)

    # Define colors based on specific words in the filename
    word_colors = {'child': 'pink', 'adult': 'orange', 'senior': 'blue'} 
    
    # Plotting
    fig, ax = plt.subplots(figsize=(20, 6))
    bar_width = 0.8
    group_spacing = 1   # Adjust the spacing between groups
    bar_positions = []
    bar_labels = []
    current_position = 0

    for i, group in enumerate(groups):
        group_data = df[df['group'] == group]
        for idx, row in group_data.iterrows():
            color = None
            for word, word_color in word_colors.items():
                if word in row['name']:
                    color = word_color
                    break
            if color is None:
                color = plt.cm.viridis(i / num_groups)  # Use default color if no specific word is found
            
            bar = ax.bar(current_position, row['speechrate(nsyll/dur)'], color=color, width=bar_width)
            bar_positions.append(current_position)
            bar_labels.append(row['name'])
            current_position += 1
        current_position += group_spacing
    
    # Create custom legend with color and speaker category correspondence
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in word_colors.values()]
    legend_labels = list(word_colors.keys())
    ax.legend(legend_handles, legend_labels, title='Color Correspondence', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Set xticks and labels
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax.set_xlabel('Name')
    ax.set_ylabel('Speaking Rate (syllables/second)')
    ax.set_title('Speaking Rate for bottleneck model')

    plt.tight_layout()
    plt.show()

# Path to your CSV file
csv_file = '../speaking_rate/SyllableNuclei_bn.txt'

# Call the function to plot speaking rates
plot_speaking_rate(csv_file)
