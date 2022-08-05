# You can go to FreddieMac website directly to download the dataset needed manually, or take advatage of this code.
# We used the source code from https://github.com/ragraw26/FreddieMac_Single_Loan_Analysis_MachineLearning/blob/09066e9c0789cf06cb66d712c3d9853afe12ab25/PART1-downloader.py#L18

# import packages
import requests
import re
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import time
import datetime
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import csv

# the dataset link in FreddieMac website
url='https://freddiemac.embs.com/FLoan/secure/auth.php'
postUrl='https://freddiemac.embs.com/FLoan/Data/download.php'

# create payload
def payloadCreation(user, passwd):
    creds={'username': user,'password': passwd}
    return creds

def assure_path_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)
def extracrtZip(s,monthlistdata,path):
    abc = tqdm(monthlistdata)
    for month in abc:
        abc.set_description("Downloading %s" % month)
        r = s.get(month)
        z = ZipFile(BytesIO(r.content)) 
        z.extractall(path)   
def getFilesFromFreddieMac(payload,st,en):
    with requests.Session() as s:
        preUrl = s.post(url, data=payload)  
        payload2={'accept': 'Yes','acceptSubmit':'Continue','action':'acceptTandC'}
        finalUrl=s.post(postUrl,payload2)
        linkhtml =finalUrl.text 
        allzipfiles=BeautifulSoup(linkhtml, "html.parser")
        ziplist=allzipfiles.find_all('td')
        sampledata=[]
        historicaldata=[]
        count=0
        slist=[]
        for i in range(int(st),int(en)+1):
            #print(i)
            slist.append(i)
        for li in ziplist:
            zipatags=li.findAll('a')
            for zipa in zipatags:
                for yr in slist:
                    if str(yr) in zipa.text:
                        if re.match('sample',zipa.text):
                            link = zipa.get('href')
                            # change to the path you want to store the files
                            Samplepath="/incorta/IncortaAnalytics/Tenants/ebs_cloud/data"+ "/FreddieMac"
                            assure_path_exists(Samplepath)
                            finallink ='https://freddiemac.embs.com/FLoan/Data/' + link
                            sampledata.append(finallink) 
        extracrtZip(s,sampledata,Samplepath)

# the 'email' and 'password' is your register account information on FreddieMac website
# we download datasets from 1999 to 2021
getFilesFromFreddieMac(payloadCreation('email', 'password'),1999,2021)

# after the process, we can find two csv files in the assigned path, which are sample_orig_files" and sample_svcg_files"