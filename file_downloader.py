from tqdm import tqdm
import getopt
import glob
import os
import requests
import sys
import zipfile

def get_urls(inputfile):
    file = open(inputfile, "r")
    urls = file.read().splitlines()
    return urls

def download_files(url):
    local_filename = url.split('/')[-1]
    print("Downloading file {0}\n".format(local_filename))
    response = requests.get(url, stream=True)

    with open(local_filename, "wb") as handle:
      for data in tqdm(response.iter_content()):
        handle.write(data)

    if zipfile.is_zipfile(local_filename):
      print("Extracting file {0}\n".format(local_filename))
      with zipfile.ZipFile(local_filename,"r") as zip_ref:
        zip_ref.extractall()
        zip_ref.close()

def main(argv):
    help_string = "file_downloader.py -i <inputfile>\n"

    if len(sys.argv) < 2:
        sys.exit(help_string)

    try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
      print(help_string)
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print(help_string)
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg

    urls = get_urls(inputfile)
    for url in urls:
        # print(url + "\n")
        download_files(url)

if __name__ == "__main__":
    main(sys.argv[1:])