import os
import matplotlib.pyplot as plt

output_dir = 'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis'
model = 'full'
file_path = 'D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/full_pitchresults_20_450.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize a dictionary to store Time and Pitch values for each filename
data_dict = {}

# Mapping of the target age colors
word_to_color = {
    'senior': 'blue',
    'adult': 'orange',
    'child': 'pink',
}
# Mapping the linestyle to original age
word_to_linestyle = {
    'c': 'dotted',
    'a': 'dashed',
    's': 'solid'
}

# Mapping the market to the original age
word_to_marker = {
    'c': '*',
    'a': '>',
    's': 'o'
}

# Loop through each line in the file and extract Time and Pitch values
for line in lines:
    if not line.startswith('Filename'):  # Skip the header
        data = line.split()
        if len(data) >= 4:
            filename = data[0]
            if data[5] != '0.0':  # Skip lines with 0.0 Pitch values
                time = float(data[4])
                pitch = float(data[5])
                if filename not in data_dict:
                    data_dict[filename] = {'time': [], 'pitch': []}
                data_dict[filename]['time'].append(time)
                data_dict[filename]['pitch'].append(pitch)
        else:
            print(f"Issue with line: {line}")

# Group data by speaker
speaker_dict = {}
for filename, data in data_dict.items():
    speaker = filename.split('_')[0]
    if speaker not in speaker_dict:
        speaker_dict[speaker] = {}
    speaker_dict[speaker][filename] = data

# Create plots for each speaker
for speaker, files in speaker_dict.items():
    plt.figure(figsize=(40, 10))
    for filename, data in files.items():
        color = 'black'  # Default color if the word is not found in the filename
        for word, col in word_to_color.items():
            if word == filename.split('_')[-2]:
                color = col
                break
        for letter, style in word_to_linestyle.items():
            if letter == filename.split('_')[0][-1]:
                linestyle = style
                break
        for letter, marker in word_to_marker.items():
            if letter == filename.split('_')[0][-1]:
                marker = marker
                break
        plt.plot(data['time'], data['pitch'], linestyle='', marker=marker, label=filename, color=color)

    plt.title(f'Speaker: {speaker}')
    plt.xlabel('Time_stamp')
    plt.ylabel('Pitch_Hz')
    plt.ylim(0.0, 2.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])

    # Save the plot with the speaker's name
    output_file = os.path.join(output_dir, f'{model}_{speaker}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

    print(f'Saved plot for speaker {speaker} at {output_file}')
    
    plt.show()
