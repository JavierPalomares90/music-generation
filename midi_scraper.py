'''
 Python class to download Midi files from a url
 Written for python3
 Usage: python3 -u <url> -d <save_dir> 
  Optional arguments:
   -r, --recursive. Recurse through a url for sublinks and 
          download all midis in the sublinks
'''
import argparse
import heapq
import re
from urllib.request import urlopen,urlretrieve
from urllib.parse import urlparse
import os
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import errno
from datetime import datetime

def get_soup(url):
    """
    Return the BeautifulSoup object for input link
    """
    # connect to the url
    website = urlopen(url)
    # get the html code
    html = website.read()
    soup = BeautifulSoup(html,features="html.parser")
    return soup

# get all the links that go to midi files
def get_all_midi_links(url):
    soup = get_soup(url)
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
        

# find all of the sub_urls in a root url
def get_all_sub_urls(source_url,depth = 0, max_depth = 5,all_urls = []):
        if depth > max_depth:
                return {}
        all_urls = []
        soup = get_soup(source_url)
        a_tags = soup.findAll("a", href=True)
        if source_url not in all_urls:
                all_urls.append(source_url)
        for a_tag in a_tags:
                href_val =  a_tag["href"]
                if "http" in href_val or "https" in href_val:
                        url = href_val
                        if url not in all_urls:
                                all_urls.append(url)
                                get_all_sub_urls(url,depth+1,max_depth,all_urls)
                elif href_val.endswith('html') or href_val.endswith('htm'):
                        url = source_url + '/' + href_val
                        if url not in all_urls:
                                all_urls.append(url)
                                get_all_sub_urls(url,depth+1,max_depth,all_urls)
                else:
                        continue


def get_midis(source_url,save_dir,recursive=False):
        if (recursive == True):
                urls = []
                max_depth = 4
                get_all_sub_urls(source_url,0,max_depth,urls)
        else:
                urls = [source_url]
        for url in urls:
                midi_links = get_all_midi_links(source_url)
                download_midis(source_url,midi_links,save_dir)


def main():
    parser = argparse.ArgumentParser(description='Download all midi files from a url')
    parser.add_argument('-u','--url',required=True,help="The url to download midis from")
    parser.add_argument('-d','--directory',required=True,help='Where to save the midis to')
    parser.add_argument('-r','--recursive',help='Recurse through the given url for midis')
    args = vars(parser.parse_args())
    source_url = args["url"]
    save_dir = args["directory"]
    recursive = False
    if(args["recursive"]):
            recursive = True
    get_midis(source_url,save_dir,recursive)

if __name__=='__main__':
    main()

