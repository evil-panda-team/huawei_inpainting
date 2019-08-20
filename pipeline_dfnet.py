# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import sys
sys.path.insert(1, './DFNet/')

from DFNet.test_vlad import Tester

# In[]:
train_path_dataset = '../DATASET_INPAINTING/train/'
test_path_dataset = '../DATASET_INPAINTING/test/'

train_file_path = train_path_dataset + 'train_mask.txt'
test_file_path = test_path_dataset + 'test_mask.txt'

path_dataset = train_path_dataset
file_path = train_file_path

# In[]:
dic = {}

with open(file_path) as f:
    content = f.readlines()

for line in content:
    values = line.split(' ', 1)
    img_id = values[0]
    value = values[1]
    val = json.loads(value)
    dic[str(img_id) + '.png'] = val

# In[]: Analyze bboxes (min max height)
h_min = 599
h_min_key = ''
h_max = 0
h_max_key = ''

for key, bboxes in dic.items():
    for bbox in bboxes:
        if bbox[0] < h_min:
            h_min = bbox[0]
            h_min_key = key
        if bbox[2] > h_max:
            h_max = bbox[2]
            h_max_key = key
            
print("Min height at {} pixel".format(h_min))
print("Max height at {} pixel".format(h_max))

# In[]: Crop 512x512 and save to temporary directory

#import tempfile
#
#with tempfile.TemporaryDirectory() as directory:
#    print('The created temporary directory is %s' % directory)

# In[]:
model = './DFNet/model/model_places2.pth'
input_size = 512
batch_size = 8
output = './DFNet/output/huawei'
tester = Tester(model, input_size, batch_size)

for k,v in tqdm(dic.items()):
    img_path = path_dataset + k
    
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[:512,:500]
    
    mask = 255*np.ones((512,512,3), dtype=np.uint8)
    for bbox in v:
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        
    cv2.imwrite("temp_img.png", img)
    cv2.imwrite("temp_mask.png", mask)
    
    y_pred = tester.inpaint("temp_img.png", "temp_mask.png")
    
    img = cv2.imread(img_path)
    img[:512,:500] = y_pred[:512,:500]
    cv2.imwrite("preds/{}".format(k), img)
    
    os.remove("temp_img.png")
    os.remove("temp_mask.png")

# In[]:
#for i, res_id in enumerate(tester.results_ids):
#    img_id = res_id.split('/')[-1]
#    img_result = tester.results[i]
#    img = cv2.imread("../DATASET_INPAINTING/test/{}".format(img_id))
#    val = dic[img_id]
#    for rect in val:
#        img[rect[0]:rect[2], rect[1]:rect[3]] = img_result[rect[0]:rect[2], rect[1]:rect[3]]
#    cv2.imwrite("result/{}".format(img_id), img)    
    
# In[]:



# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




