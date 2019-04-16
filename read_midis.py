'''
 Python class to load the download midi files
 Written for python3
 Usage: python3 -d <midi_file_directory> 
'''
import argparse

def load_midis(dir):
    # TODO: complete impl
    0

def main():
    parser = argparse.ArgumentParser(description='Download all midi files from a url')
    parser.add_argument('-d','--directory',required=True,help='Where to save the midis to')
    args = vars(parser.parse_args())
    dir = args["directory"]
    load_midis(dir)



if __name__=='__main__':
    main()
