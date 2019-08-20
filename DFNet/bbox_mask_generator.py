# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

# In[]:
bbox_w_max = 127
bbox_w_min = 80
bbox_h_min = 80
bbox_h_max = 127

height=512
width=512
mask = np.ones((height, width))

bboxes_num = np.random.randint(1,5)

for i in range(bboxes_num):
    x = np.random.randint(0,512)
    y = np.random.randint(0,512)
    w = np.random.randint(bbox_w_min,bbox_w_max)
    h = np.random.randint(bbox_h_min,bbox_h_max)
    print(x,y,w,h)
    cv2.rectangle(mask, (x-bbox_w_min,y-bbox_h_min), (x-bbox_w_min+w,y-bbox_h_min+h), 0., -1)
plt.imshow(mask)

# In[]:


# In[]:




# In[]:




# In[]:




# In[]:




