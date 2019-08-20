# -*- coding: utf-8 -*-

import cv2
import pickle
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt

# In[]:
with open('train_dict.pkl', 'rb') as f:
    train_dict = pickle.load(f)

# In[]:
train_data_dir = '../DATASET_INPAINTING/train/'

# In[]:
#for k,v in train_dict.items():
#    img_gt_path = train_data_dir + str(k) + '_gt.png'
#    img_gt = cv2.imread(img_gt_path)
#    for i, bbox in enumerate(v):
#        img = img_gt.copy()
#        img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
#        cv2.imwrite("../DATASET_INPAINTING/trainA/{}_{}.png".format(k, i), img)        
#        copyfile(img_gt_path, "../DATASET_INPAINTING/trainB/{}_{}.png".format(k, i))
#    print(k)

# In[]:
for k,v in train_dict.items():
    img_gt_path = train_data_dir + str(k) + '_gt.png'
    img_gt = cv2.imread(img_gt_path)
    for i, bbox in enumerate(v):
        img = img_gt.copy()
        img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        cv2.imwrite("../DATASET_INPAINTING/train_pix2pix/{}_{}.png".format(k, i), np.hstack((img, img_gt)))        
    print(k)

# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




