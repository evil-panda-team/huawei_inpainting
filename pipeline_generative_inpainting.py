# -*- coding: utf-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import tensorflow as tf
import neuralgym as ng
from tqdm import tqdm
from classical import classical_or_dfnet, perimeter, classical

import sys
sys.path.insert(1, './generative_inpainting/')

from inpaint_model import InpaintCAModel

# In[]:
train_path_dataset = '../DATASET_INPAINTING/train/'
test_path_dataset = '../DATASET_INPAINTING/test/'

train_file_path = train_path_dataset + 'train_mask.txt'
test_file_path = test_path_dataset + 'test_mask.txt'

file_path = test_file_path
path_dataset = test_path_dataset

# In[]:
dic = {}

with open(file_path) as f:
    content = f.readlines()

for line in tqdm(content):
    values = line.split(' ', 1)
    img_id = values[0]
    value = values[1]
    val = json.loads(value)
    dic[str(img_id) + '.png'] = val

# In[]:
img_dir = test_path_dataset
checkpoint_dir = 'generative_inpainting/model_logs/release_places2_256/'

# In[]:
model = InpaintCAModel()

# In[]:
for k,v in tqdm(dic.items()):
    img_path = path_dataset + k
    
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[:512,:500]
    
    mask = np.zeros((512,512,3), dtype=np.uint8)
    for bbox in v:
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 255
    
    assert img.shape == mask.shape
    
    h, w, _ = img.shape
    grid = 8
    img = img[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    
    img = np.expand_dims(img, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([img, mask], axis=2)
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        result = sess.run(output)
        
        img = cv2.imread(img_path)
        y_pred = result[0][:, :500, ::-1]
        img[:512,:500] = y_pred[:512,:500]
        
        # BLURRING
        img_blurred = img.copy()
        
        #Postprocessing
        img_blurred = cv2.medianBlur(img_blurred, 13)
        img_blurred = cv2.GaussianBlur(img_blurred, (13, 13), 0)
        img_blurred = cv2.blur(img_blurred, (9, 9))
        img_blurred = cv2.bilateralFilter(img_blurred, 17, 75, 75)
    
        for bbox in v:
            img[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        cv2.imwrite('result/{}'.format(k), img)
    tf.reset_default_graph()

# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




