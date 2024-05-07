from utils.generate_lab_files import create_labs

main_directory = 'FilteredCV17' # Folder 
file_mapping = 'raw_data/CV_7/gender_balanced_CV_7.txt' # Text

create_labs(main_directory, file_mapping)