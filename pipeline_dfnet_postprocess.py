# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import json
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(1, './DFNet/')

from DFNet.test_vlad import Tester

# In[]:
path_dataset = '../DATASET_INPAINTING/test/'

# In []:
def generate_mask(img, minsize):
    min_size = 2000
    thresh = cv2.inRange(img, (255,255,255), (255,255,255))
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    mask = 255*np.ones((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask[output == i + 1] = 0
        
    return mask

# In[]:
images = glob(path_dataset + '*.png')
images.sort()
#
#i = 0
##
#img_path = images[i]
##
#img = cv2.imread(img_path)
#mask = generate_mask(img, minsize = 2000)
#
#contours,hierarchy = cv2.findContours(mask, 1, 2)
#cnt = contours[0]
##        
#epsilon = 0.1*cv2.arcLength(cnt,True)
#approx = cv2.approxPolyDP(cnt,epsilon,True)
#    
#plt.imshow(mask)

# In[]:
model = './DFNet/model/model_places2.pth'
input_size = 512
batch_size = 8
tester = Tester(model, input_size, batch_size)

for img_path in tqdm(images):
    
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[:512,:500]
    
    mask = np.zeros((512,512), dtype=np.uint8)
    mask[:512,:500] = generate_mask(img, minsize = 2000)[:512,:500]
    
    y_pred = tester.inpaint(img, mask)
    
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[:512,:500]
    img[:512,:500] = y_pred[:512,:500]

    # BLURRING
    img_blurred = img.copy()
    
    #Postprocessing
    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
    img_blurred = cv2.medianBlur(img_blurred, 11)
    
    print(img_blurred.shape)
    print(img.shape)
    print(mask.shape)
    mask = mask.astype(bool)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not mask[i,j]:
                img[i,j] = img_blurred[i,j]
                
    result = cv2.imread(img_path)
    result[:512,:500] = img[:512,:500]
                
    cv2.imwrite("result/{}".format(img_path.split('/')[-1]), result)

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




