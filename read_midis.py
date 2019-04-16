'''
 Python class to load the download midi files
 Written for python3
 Usage: python3 -d <midi_file_directory> 
'''
import argparse
import os
from pymidifile import *

def get_midi_paths(dir):
    # Find all the midi files in the directory
    paths = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if file.endswith(".mid"):
                paths.append(os.join(root,file))
    return paths

def get_pandas_dataframes(paths):
    for path in paths:
        matrix = mid_to_matrix(path)


def main():
    parser = argparse.ArgumentParser(description='Download all midi files from a url')
    parser.add_argument('-d','--directory',required=True,help='Where to save the midis to')
    args = vars(parser.parse_args())
    dir = args["directory"]
    paths = get_midi_paths(dir)
    get_pandas_dataframes(paths)



if __name__=='__main__':
    main()
