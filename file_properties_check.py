import soundfile as sf
import os

folder = 'agingTTS'
subfolder_path_list = os.listdir(folder)
target_subtype = 'PCM_16'

for subfolder in subfolder_path_list:
    subfolder_path = os.path.join(folder, subfolder)
    print(f'Checking speaker {subfolder}')
    
    for file in os.listdir(subfolder_path):
        if file.split('.')[-1] == 'wav':
            audio_file = os.path.join(subfolder_path, file)
            # Read the audio file
            data, samplerate = sf.read(audio_file)
            with sf.SoundFile(audio_file) as f:
                print(f'File: {file}, {f.subtype}')
                # Convert and save if subtype is not the target subtype
                if f.subtype != target_subtype:
                    temp_file = audio_file + '.temp.wav'
                    sf.write(temp_file, data, samplerate, subtype=target_subtype)
                    os.replace(temp_file, audio_file)
                    print(f'Converted {file} to {target_subtype}')
