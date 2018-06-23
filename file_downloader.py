import getopt
import os.path
import sys
import urllib.request
import zipfile

class Food_11_Downloder():

  def get_urls(self, inputfile):
    file = open(inputfile, "r")
    urls = file.read().splitlines()
    return urls

  def download_files(self, url):
    local_filename = url.split('/')[-1]

    if not os.path.exists(local_filename):
      print("Downloading file {0}\n".format(local_filename))
      urllib.request.urlretrieve(url, local_filename)

      if zipfile.is_zipfile(local_filename):
        print("Extracting file {0}\n".format(local_filename))
        with zipfile.ZipFile(local_filename,"r") as zip_ref:
          zip_ref.extractall()
          zip_ref.close()

  @staticmethod
  def may_be_download_food_11_dependency(file = None):
    downloder = Food_11_Downloder()
    if not file:
      file = "external_downloads_url.txt"
      urls = downloder.get_urls(file)
    for url in urls:
      downloder.download_files(url)


  def test_main(self, argv):
    inputfile = ""
    help_string = "file_downloader.py -i <inputfile>\n"
    default_file = "external_downloads_url.txt"

    if len(sys.argv) < 2:
      self.download()
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
           self.download(arg)

if __name__ == "__main__":
    downloder = Food_11_Downloder()
    downloder.test_main(sys.argv[1:])