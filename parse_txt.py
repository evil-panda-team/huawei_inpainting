# -*- coding: utf-8 -*-

import numpy as np
import pickle

# In[]:
# Attention to last '\n'
train_data = '../DATASET_INPAINTING/train/train_mask.txt'
train_dict = {}
for line in open(train_data,'r'):
    line_parts = line.split(' ')
    img_id = line_parts[0]
    bboxes = ''.join(line_parts[1:])
    bboxes = bboxes[1:-2]
    bboxes = bboxes.split('[')[1:]
    bboxes_int = []
    for bb in bboxes:
        bbox_string = bb.split(']')[0].split(',')
        bbox_int = [int(bbox_string[0]), int(bbox_string[1]), int(bbox_string[2]), int(bbox_string[3])]
        bboxes_int.append(bbox_int)      
    train_dict[int(img_id)] = np.array(bboxes_int)

# In[]:
with open('train_dict.pkl', 'wb') as f:
    pickle.dump(train_dict, f)
    
# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:




# In[]:



