import getopt
import glob
import os.path
import sys
import urllib.request
import zipfile

def get_urls(inputfile):
    file = open(inputfile, "r")
    urls = file.read().splitlines()
    return urls

def download_files(url):
    local_filename = url.split('/')[-1]

    if not os.path.exists(local_filename):
      print("Downloading file {0}\n".format(local_filename))
      urllib.request.retrieve(url, local_filename)

      if zipfile.is_zipfile(local_filename):
        print("Extracting file {0}\n".format(local_filename))
        with zipfile.ZipFile(local_filename,"r") as zip_ref:
          zip_ref.extractall()
          zip_ref.close()

def main(argv):
    inputfile = ""
    help_string = "file_downloader.py -i <inputfile>\n"
    default_file = "external_downloads_url.txt"

    if len(sys.argv) < 2:
      input_file = default_file
    else:
      try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
      except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
      for opt, arg in opts:
        if opt == '-h':
           sys.exit(help_string)
        elif opt in ("-i", "--ifile"):
           inputfile = arg

    urls = get_urls(inputfile)
    for url in urls:
        # print(url + "\n")
        download_files(url)

if __name__ == "__main__":
    main(sys.argv[1:])