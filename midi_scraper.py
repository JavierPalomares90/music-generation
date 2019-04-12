# Python class to download Midi files from a url
# Written for python3
# Usage:
import argparse
import heapq
import re
from urllib.request import urlopen
from urllib.parse import urlparse
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup

def get_all_links(url):
    # connect to the url
    website = urlopen(url)
    # get the html code
    html = website.read()
    soup = BeautifulSoup(html)

    links = soup.find_all('mid')

    return links

# get all the links that go to midi files
def get_midi_links(links):
    midis = []
    pattern = re.compile(".*\.mid")
    for tuple in links:
        link = tuple[0]
        matches = re.match(pattern,link)
        if matches:
            midis.append(link)
    return midis 

def download_midis(midis,save_dir):
    for midi_link in midis:
        midi = urlopen(midi_link)
        output = urlparse(midi_link)
        print("Downloading {}".format(midi_link))
        file_name = os.path.basename[output[2]]
        path = save_dir + file_name
        f = open(path,'wb+')
        f.write(midi.read())
        f.close()


def main():
    parser = argparse.ArgumentParser(description='Download all midi files from a url')
    parser.add_argument('-u','--url',required=True,help="The url to download midis from")
    parser.add_argument('-d','--directory',required=True,help='Where to save the midis to')
    args = vars(parser.parse_args())
    source_url = args["url"]
    save_dir = args["directory"]
    links = get_all_links(source_url)
    midis = get_midi_links(links)
    download_midis(midis,save_dir)


if __name__=='__main__':
    main()

