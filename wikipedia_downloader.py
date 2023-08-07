from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import progressbar
import shutil
import subprocess
import bz2
import os
import re

class DownloadPBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar= progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()



class WikiDownloader:

    def __init__(self, lang, out_path) -> None:
        wlang = lang + "wiki"
        self.XML_FILE = "{}wiki-latest-pages-articles.xml".format(lang)
        self.BZ_FILE = "{}.bz2".format(self.XML_FILE)
        self.BASE_WIKI_PATH = "https://dumps.wikimedia.org/{}/latest/{}".format(wlang, self.BZ_FILE)
        self.OUT_PATH = out_path
        self.WIKI_NAME = wlang

    def download_wiki(self, override_path = None):
        if override_path:
            self.OUT_PATH = override_path

        if not os.path.isdir(self.OUT_PATH):
            os.mkdir(self.OUT_PATH)
        
        out_xml_file = "{}/{}".format(self.OUT_PATH, self.XML_FILE)
        out_bz_file = out_xml_file + ".bz2"
        if os.path.isfile(out_xml_file):
            print("El dump de la wiki ya se encuentra descargado...")
            return

        if not os.path.isfile(out_bz_file):
            out_path, status = urlretrieve(self.BASE_WIKI_PATH, out_bz_file, DownloadPBar())
            print("File Downloaded...")

        else:
            print("BZ2 File already in path")

    def decompress_bz_file(self):
        print("Unziping...")
        bz_path = self.OUT_PATH + "/" + self.BZ_FILE
        xml_path = self.OUT_PATH + "/" + self.XML_FILE
        zipfile = bz2.BZ2File(bz_path)
        zipfile.seek(0)
        data = zipfile.read()
        open(xml_path, 'wb').write(data)
        print("Finish inflating")

    def extract_wiki(self):
        print("Extracting...")
        xml_path = self.OUT_PATH + "/" + self.XML_FILE
        print(xml_path)
        std_out = subprocess.run([ "python", "-m" "wikiextractor.WikiExtractor", 
                        "--processes", 4, "--no_templates", "--min_text_length", 2000, "--filter_disambing_pages",
                         "--log_file", "log", "-b", "1GB", "-q", xml_path ])

        #shutil.move("{}/text/AA/wiki_00".format(self.OUT_PATH), "{}/{}".format(self.OUT_PATH, self.WIKI_NAME))
        #shutil.rmtree("{}/text".format(self.OUT_PATH))

    def split_wiki(self, override_path = None, lang = "es"):
        if override_path:
            self.OUT_PATH = override_path
        
        docs_path = self.OUT_PATH + "/docs"
        if os.path.isdir(docs_path):
            print(f"{docs_path} already exists; not splitting")
            return docs_path
        
        os.makedirs(docs_path)
        title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
        lines = open(self.OUT_PATH + "/wiki_00", "r").readlines()
        
        f = None

        for i, l in enumerate(lines):
            if i % 100000 == 0: print(i)
            if l.startswith('<doc id="'):
                title = title_re.findall(l)[0].replace('/','_')
                if "desambiguación" in title or len(title)>150: continue
                if f: f.close()
                file_path = docs_path + "/" + title + ".txt"
                f = open( file_path, "w" )
                f.write(l)
            else: f.write(l)
        f.close()
        return docs_path
    
    def extract_data_from_file(self, file_name):
        content_of_file = ""
        doc_re = re.compile(rf'([\w\W]*)<\/doc>') # delete </doc>
        with open("es_wiki/docs/" + file_name, "r", encoding="utf8") as f:
            f.readline()
            content_of_file = f.read()
            content_of_file = doc_re.findall(content_of_file)[0].strip()
            #content_of_file += "\n"
        return content_of_file

    def get_one_clean_file(self, dest, lang = "es"):
        fname = f'all_texts_{lang}wiki.txt'
        files_to_concat = os.listdir(dest)
        #files_to_concat = files_to_concat[0:30]
        total_files = len(files_to_concat)
        final_path = "es_wiki"
        with open ( "{}/{}".format(final_path, fname),  'w') as fp: 
            for i, file in enumerate(files_to_concat):
                if not (i % 1000): 
                    print("Parsing file #{} of {}".format(i, total_files))
                fp.write(self.extract_data_from_file(file))


    def get_one_clean_csv_file(self, dest, lang = "es"):
        fname = f'all_texts_{lang}wiki.csv'
        files_to_concat = os.listdir(dest)
        files_to_concat = files_to_concat[0:200000]
        total_files = len(files_to_concat)
        final_path = "es_wiki"
        #list_of_contents = [ ]
        df = pd.DataFrame([], columns=["text"])
        for i, file in enumerate(files_to_concat):
            if not (i % 1000): 
                print("Parsing file #{} of {}".format(i, total_files))
            #list_of_contents.append(self.extract_data_from_file(file))
            df.loc[len(df)] = self.extract_data_from_file(file)
        
        print("Exporting DF")
        df.to_csv("{}/{}".format(final_path, fname), index=False)  
