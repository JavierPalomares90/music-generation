# Python class to download Midi files from a url
# Written for python3
# Usage:
import argparse
import heapq
import re
from urllib.request import urlopen,urlretrieve
from urllib.parse import urlparse
import os
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup

# get all the links that go to midi files
def get_all_midi_links(url):
    # connect to the url
    website = urlopen(url)
    # get the html code
    html = website.read()
    soup = BeautifulSoup(html,features="html.parser")
    links = soup.findAll(href=re.compile(r"\.mid$"))
    return links

def download_midis(url,midis,save_dir):
    for midi_link in midis:
        file_name = midi_link.get('href')
        dl_link = url + '/' + file_name
        midi = urlopen(dl_link)
        output = urlparse(dl_link)
        print("Downloading {}".format(dl_link))
        path = os.path.join(save_dir,file_name)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
                try:
                        os.makedirs(dir_path)
                except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                                raise
                
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
    midi_links = get_all_midi_links(source_url)
    download_midis(source_url,midi_links,save_dir)

if __name__=='__main__':
    main()

