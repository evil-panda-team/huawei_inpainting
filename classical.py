# -*- coding: utf-8 -*-

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# In[]:
train_path_dataset = '../DATASET_INPAINTING/train/'
test_path_dataset = '../DATASET_INPAINTING/test/'
train_file_path = train_path_dataset + 'train_mask.txt'
test_file_path = test_path_dataset + 'test_mask.txt'

# In[]: PIPELINE
def perimeter(img, rect, neighborhood, gray):
    n = neighborhood
    img_copy = img.copy()

    if gray:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        
    temp = img_copy[rect[0]-n:rect[2]+n, rect[1]-n:rect[1]]
    if gray:
        bbox = temp.reshape(temp.shape[0]*temp.shape[1])
    else:
        bbox = temp.reshape(temp.shape[0]*temp.shape[1], 3)
        
    temp = img_copy[rect[0]-n:rect[2]+n, rect[3]:rect[3]+n]
    if gray:
        bbox = np.concatenate((bbox, temp.reshape(temp.shape[0]*temp.shape[1])), axis = 0)
    else:
        bbox = np.concatenate((bbox, temp.reshape(temp.shape[0]*temp.shape[1], 3)), axis = 0)
        
    temp = img_copy[rect[0]-n:rect[0], rect[1]:rect[3]]
    if gray:
        bbox = np.concatenate((bbox, temp.reshape(temp.shape[0]*temp.shape[1])), axis = 0)
    else:
        bbox = np.concatenate((bbox, temp.reshape(temp.shape[0]*temp.shape[1], 3)), axis = 0)

    temp = img_copy[rect[2]:rect[2]+n, rect[1]:rect[3]]
    if gray:
        bbox = np.concatenate((bbox, temp.reshape(temp.shape[0]*temp.shape[1])), axis = 0)
    else:
        bbox = np.concatenate((bbox, temp.reshape(temp.shape[0]*temp.shape[1], 3)), axis = 0)

    if gray:
        return bbox.flatten()
    else:
        return bbox


def classical_or_dfnet(bbox, thresh, histwidth):
    thresh = len(bbox)*thresh
    b = Counter(bbox)
    freq = b.most_common()
    most = freq[0][0]
    h = np.histogram(bbox, bins = 1, range = [most-histwidth, most+histwidth])
    if h[0][0] >= thresh:
        return "classical"
    else:
        return "dfnet"

def classical(img, rect, p):
    avg_color = p.mean(axis=0).astype(np.uint8)
    img[rect[0]:rect[2], rect[1]:rect[3]] = avg_color

# In[]:
thresh = 0
histwidth = 5
neighborhood = 3

file_path = test_file_path
path_dataset = test_path_dataset

with open(file_path) as f:
    content = f.readlines()
    for line in tqdm(content):
        values = line.split(' ', 1)
        img_id = values[0]
        value = values[1]
        val = json.loads(value) 

        mask = 255 * np.ones((600,500), dtype=np.uint8)
        img = cv2.imread(path_dataset+img_id+'.png')

        for rect in val:
            p_gray = perimeter(img, rect, neighborhood = neighborhood, gray = True)
            decision = classical_or_dfnet(p_gray, thresh = thresh, histwidth = histwidth)
            
            if decision == "classical":
                p = perimeter(img, rect, neighborhood = 3, gray = False)
                classical(img, rect, p)
                cv2.imwrite("results/{}.png".format(img_id), img)

# In[]:




# In[]:




# In[]:




# In[]:




# In[]:





