# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(1, './DFNet/')

from DFNet.test_vlad import Tester

# In[]:
train_path_dataset = './DATASET_INPAINTING/train/'
test_path_dataset = '../DATASET_INPAINTING/test/'

train_file_path = train_path_dataset + 'train_mask.txt'
test_file_path = test_path_dataset + 'test_mask.txt'

path_dataset = test_path_dataset
file_path = test_file_path

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

# In[]:
model = './DFNet/model/model_places2.pth'
input_size = 512
batch_size = 8
tester = Tester(model, input_size, batch_size, 'cpu')

for k,v in tqdm(dic.items()):
    img_path = path_dataset + k
    
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[:512,:500]
    
    mask = 255*np.ones((512,512,1), dtype=np.uint8)
    for bbox in v:
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
    
    y_pred = tester.inpaint(img, mask)
    
    img = cv2.imread(img_path)
    img[:512,:500] = y_pred[:512,:500]
    
    # BLURRING
    img_blurred = img.copy()
    
    #Postprocessing
    img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, 1)
    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
    img_blurred = cv2.medianBlur(img_blurred, 11)

    for bbox in v:
        img[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
    cv2.imwrite("result2/{}".format(k), img)

# In[]:

    
# In[]:



# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




