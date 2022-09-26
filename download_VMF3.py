# import os
# from ftplib import FTP

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import subprocess, os
from os.path import expanduser
####    PARAMETERS:

vmf_ftp_url = 'https://vmf.geo.tuwien.ac.at/trop_products/GRID/1x1/VMF3/VMF3_OP/2020/'

months = ['01','07']


##################3

def create_folder_ifnotexists(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)


homepath = expanduser("~")

outfolder = os.path.join(homepath,'data/VMF3')


create_folder_ifnotexists(outfolder)

for month in months:
    monthpath = os.path.join(outfolder,month)
    create_folder_ifnotexists(monthpath)



response = requests.get(vmf_ftp_url)

soup = BeautifulSoup(response.text) 

for link in soup.findAll('a'):
    linkname = link.get('href')

    if not '/' in linkname:
        if 'VMF3_' in linkname:

            for month in months:
            
                if linkname[9:11] == month:
                    file_outpath = os.path.join(outfolder,month,linkname)
                    full_url = urljoin(vmf_ftp_url,linkname)
                    subprocess.run(f'wget -O "{file_outpath}" "{full_url}"',shell=True)