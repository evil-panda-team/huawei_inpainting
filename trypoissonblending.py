# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from glob import glob
from poissonblending import prepare_mask, blend
import math

# In[]:
train_path_dataset = '../DATASET_INPAINTING/train/'
train_gt_path_dataset = '../DATASET_INPAINTING/gt/'
test_path_dataset = '../DATASET_INPAINTING/test/'

train_file_path = '../DATASET_INPAINTING/train/train_mask.txt'
test_file_path = test_path_dataset + 'test_mask.txt'

path_dataset = train_path_dataset
file_path = train_file_path

preds_path = 'preds/train/'

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
    
# In[]:
def psnr(x, y):
   mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
   if mse == 0:
       return 100
   return 20 * math.log10(255.0 / math.sqrt(mse))

# In[]:
for k,v in dic.items():
    img = cv2.cvtColor(cv2.imread(train_path_dataset + k), cv2.COLOR_BGR2RGB)
    img_gt = cv2.cvtColor(cv2.imread(train_gt_path_dataset + k), cv2.COLOR_BGR2RGB)
    img_pred = cv2.cvtColor(cv2.imread(preds_path + k), cv2.COLOR_BGR2RGB)
    
    mask = np.zeros((600,500,3), dtype=np.uint8)
    for bbox in v:
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255
        
    out = blend(img.copy(), img_pred.copy(), mask.copy(), offset=(0, 0))
    print("BEFORE: {}, AFTER: {}".format(psnr(img_gt,img_pred), psnr(img_gt,out)))
#plt.imshow(img)
#psnr(img_pred, img_gt)
    
    
# In[]:



# In[]:


    
    
# In[]:
    
    
    
    
# In[]:

    
    
    
# In[]:
    
    
    
    
    