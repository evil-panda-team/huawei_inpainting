# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import json
import cv2
from tqdm import tqdm
import zipfile
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, './DFNet/')

from DFNet.test_vlad import Tester

# In[]:
def psnr(x, y):
   mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
   if mse == 0:
       return 100
   return 20 * math.log10(255.0 / math.sqrt(mse))

# In[]:
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default = '../DATASET_INPAINTING/train/')
args = parser.parse_args()

# In[]:
#dataset_path = '../DATASET_INPAINTING/test_final/'
#txt_path = 'test_final_mask.txt'
dataset_path = args.img_dir
txt_path = dataset_path + 'train_mask.txt'

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

# In[]

avg_psnr = 0

n = 800
i = 0

for k,v in tqdm(dic.items()):
    
    if i >= n:
        break
    
# In[]
    img_path = dataset_path + k
    
    mask = 255*np.ones((600,512), dtype=np.uint8)
    for bbox in v:
#        bb2 = 600 if bbox[2] > 600 else bbox[2]
#        bb3 = 500 if bbox[3] > 500 else bbox[3]
#        mask[bbox[0]:bb2, bbox[1]:bb3] = 0
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        
    #   PADDING VERSION
    pad_right = cv2.copyMakeBorder(cv2.imread(img_path),0,0,0,12,cv2.BORDER_REPLICATE)
    img = pad_right[:512,:512].copy()
    y_pred = tester.inpaint(img, mask[:512,:512])
    
    # Concatenate
    img = cv2.imread(img_path)
    img[:512,:500] = y_pred[:512,:500]
    
    # Postprocessing
    img_blurred = img.copy()
#    img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, 1)
    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
    img_blurred = cv2.medianBlur(img_blurred, 11)

    for bbox in v:
        img[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
    score = psnr(cv2.imread(img_path.replace('train','gt')), img)
    print(score)
    
# In[]
    
    avg_psnr = avg_psnr + score/n
    i = i + 1
    
print("AVERAGE: {}".format(avg_psnr))

#
# Bare 30.38167518624206
# Gaussian + median 31.522442211210016
# Gaussian + median (bboxes) 31.176960969799925
# Gaussian + median + padding (replicate) 31.445780553353895