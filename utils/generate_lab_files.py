import os
from tqdm import tqdm

def create_labs(main_directory, file_mapping):
  with open(file_mapping, encoding="utf-8") as f:
      lines = f.readlines()
      print(lines)
  
  for line in tqdm(lines[1:]):
      parts = line.strip().split("\t")
      folder_name = parts[0]
      audio_name = parts[1]
      audio_name_extension = audio_name.split('.')[-1]
      audio_name = audio_name.replace(f'.{audio_name_extension}', '')
      transcription = parts[2]
      
      lab_file_path = os.path.join(main_directory, folder_name, "{}.lab".format(audio_name))
      with open(lab_file_path, "w", encoding="utf-8") as f1:
          f1.write(transcription)