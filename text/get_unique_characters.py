import string
import argparse

def get_unique_chars(file_path):

    '''Provides a set with all the unique characters found in the 
    txt file given as argument, excluding whitespaces, tab and newlines.
    
    Returns a set of unique characters'''
    
    with open(file_path, 'r') as f:
        text = f.readlines()
        
    ipa_chars_list = []  
      
    for line in text:    
        ipa_word = line.split('\t')[-1]
        ipa_chars = ipa_word.replace('\n', '').split(' ')
        for c in ipa_chars:
          ipa_chars_list.append(c)
        
    full_unique_chars = set(ipa_chars_list)
    unique_chars = set()
    
    for char in full_unique_chars:
    
      if char not in string.punctuation and char not in string.digits:
          unique_chars.add(char)
            
    return unique_chars
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        required=True,
        help="Path to the file with trascriptions",
    )
       
    args = parser.parse_args()
    
    get_unique_chars(args)