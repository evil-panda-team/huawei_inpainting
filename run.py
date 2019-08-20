#! /usr/bin/env python3

import os
import argparse
import numpy as np
import json
import cv2
from tqdm import tqdm
import zipfile

import sys
sys.path.insert(1, './DFNet/')

from DFNet.test_vlad import Tester

# In[]:
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default = '../DATASET_INPAINTING/test_final/')
args = parser.parse_args()

# In[]:
dataset_path = args.img_dir
txt_path = 'test_final.txt'

# In[]:
dic = {}

with open(txt_path) as f:
    content = f.readlines()

for line in content:
    values = line.split(' ', 1)
    img_id = values[0]
    value = values[1]
    val = json.loads(value)
    dic[str(img_id) + '.png'] = val

# In[]:
model = './DFNet/model/model_places2.pth'
input_size = 512
batch_size = 1
tester = Tester(model, input_size, batch_size)
#tester = Tester(model, input_size, batch_size, 'cpu')

zf = zipfile.ZipFile('result.zip', mode='w')

for k,v in tqdm(dic.items()):
    
    img_path = dataset_path + k
    
    pad = False
    for bbox in v:
        bb3 = bbox[3]
        if bb3 >= 500 or bbox[0]==0 or bbox[1]==0:
            pad = True
        
    if pad:
        mask = 255*np.ones((600,500), dtype=np.uint8)
        for bbox in v:
            bb3 = bbox[3]
            mask[bbox[0]:bbox[2], bbox[1]:bb3] = 0
        
        mask = cv2.copyMakeBorder(mask,3,0,6,6,cv2.BORDER_REPLICATE)
        pad_right = cv2.copyMakeBorder(cv2.imread(img_path),3,0,6,6,cv2.BORDER_REPLICATE)
        img = pad_right[:512,:512].copy()
    else:
        mask = 255*np.ones((600,512), dtype=np.uint8)
        for bbox in v:
            bb3 = bbox[3]
            if bb3 >= 500:
                bb3 = 550
            mask[bbox[0]:bbox[2], bbox[1]:bb3] = 0
        img = np.zeros((512,512,3), dtype=np.uint8)
        img[:512,:500] = cv2.imread(img_path)[:512,:500]
        
    y_pred = tester.inpaint(img, mask[:512,:512])
    
    # Concatenate
    img = cv2.imread(img_path)
    if pad:
        img[:500,:500] = y_pred[3:503,6:506]
    else:
        img[:512,:500] = y_pred[:512,:500]
    
    # Postprocessing
    img_blurred = img.copy()
    img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, 55)
    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
    img_blurred = cv2.medianBlur(img_blurred, 11)

    for bbox in v:
        img[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
    cv2.imwrite("{}".format(k), img)
    zf.write(k)
    os.remove(k)
    
zf.close()