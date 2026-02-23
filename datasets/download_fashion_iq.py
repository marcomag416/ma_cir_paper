import os
import urllib.request
import argparse

"""
Adapted from https://github.com/yanbeic/VAL/blob/master/download_fashion_iq.py

download url files from https://github.com/hongwang600/fashion-iq-metadata/tree/master/image_url
"""

### This is a script to download the fashion-iq dataset

# python download_image_data.py --split=0
# python download_image_data.py --split=1
# python download_image_data.py --split=2

parser = argparse.ArgumentParser(description="Download Fashion IQ images.")
parser.add_argument('--split', type=int, default=0, help='split index: 0=dress, 1=shirt, 2=toptee')
args = parser.parse_args()

readpath = ['data/metadata/fashion_iq/image_url/asin2url.dress.txt', \
            'data/metadata/fashion_iq/image_url/asin2url.shirt.txt', \
            'data/metadata/fashion_iq/image_url/asin2url.toptee.txt']

savepath = ['data/images/fashion_iq/dress', \
            'data/images/fashion_iq/shirt', \
            'data/images/fashion_iq/toptee']

missing_file = ['data/logs/fashion_iq/missing_dress.log', \
                'data/logs/fashion_iq/missing_shirt.log', \
                'data/logs/fashion_iq/missing_toptee.log']

k = args.split

if not os.path.exists(savepath[k]):
  os.makedirs(savepath[k])

os.makedirs(os.path.dirname(missing_file[k]), exist_ok=True)

with open(missing_file[k], 'a') as f:
  missing = 0
  file = open(readpath[k], "r")
  lines = file.readlines()
  print(len(lines))
  for i in range(len(lines)):
    try:
      line = lines[i].replace('\n','').split(' \t ')
      url = line[1]
      imgpath = os.path.join(savepath[k], line[0]+'.jpg')
      urllib.request.urlretrieve(url, imgpath)
    except:
      missing += 1
      f.write(imgpath+"\n")
      print(imgpath)
      pass

print("missing %d." % missing)