import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from collections import Counter
        
# In[]:
path_dataset = '../DATASET_INPAINTING/train/'
file_path = path_dataset + 'train_mask.txt'

# In[]:
with open(file_path) as f:
    content = f.readlines()
    for line in content:
        values = line.split(' ', 1)
        img_id = values[0]
        value = values[1]
        val = json.loads(value) 

        mask = 255 * np.ones((600,500), dtype=np.uint8)
        img = cv2.imread(path_dataset+img_id+'.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for rect in val:
            bbox = img[rect[0]:rect[2], rect[1]:rect[3]]
#            mask[rect[0]:rect[2],rect[1]:rect[3]] = 0



# In[]:
n = 3
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
bbox = gray[rect[0]-n:rect[2]+n, rect[1]-n:rect[3]+n]
plt.imshow(bbox, cmap='gray')

# In[]:
bbox = gray[rect[0]-n:rect[2]+n, rect[1]-n:rect[1]].flatten()
bbox = np.append(bbox, gray[rect[0]-n:rect[2]+n, rect[3]:rect[3]+n].flatten())
bbox = np.append(bbox, gray[rect[0]:rect[2], rect[1]-n:rect[1]].flatten())
bbox = np.append(bbox, gray[rect[0]:rect[2], rect[3]:rect[3]+n].flatten())

# In[]:
#h = np.histogram(bbox, bins = 256, range = [0, 256])
b = np.bincount(bbox, minlength=256)

# In[]:
hist = cv2.calcHist([bbox],[0],None,[256],[0,256])
#plt.hist(bbox,256,[0,256])
#plt.show()
plt.plot(hist)

# In[]:
thresh = len(bbox)*0.8
b = Counter(bbox)
freq = b.most_common()
most = freq[0][0]
h = np.histogram(bbox, bins = 1, range = [most-5, most+5])
if h[0][0] >= thresh:
    print("Classic")
else:
    print("DFGAN")

# In[]: PIPELINE
def perimeter(img, rect, neighborhood, gray):
    n = neighborhood
    img_copy = img.copy()

    if gray == True:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    plt.imshow(img_copy)
        
    bbox = img_copy[rect[0]-n:rect[2]+n, rect[1]-n:rect[1]]
    bbox = np.concatenate((bbox, img_copy[rect[0]-n:rect[2]+n, rect[3]:rect[3]+n]), axis=0)
    bbox = np.concatenate((bbox, img_copy[rect[0]:rect[2], rect[1]-n:rect[1]]), axis=0)
    bbox = np.concatenate((bbox, img_copy[rect[0]:rect[2], rect[3]:rect[3]+n]), axis=0)
    
    if gray == True:
        return bbox.flatten()
    else:
        return bbox.reshape((bbox.shape[0]*bbox.shape[1], 3))


def classical_or_dfnet(bbox, thresh, histwidth):
    thresh = len(bbox)*thresh
    b = Counter(bbox)
    freq = b.most_common()
    most = freq[0][0]
    h = np.histogram(bbox, bins = 1, range = [most-histwidth, most+histwidth])
    print(h[0][0]/len(bbox))
    if h[0][0] >= thresh:
        return "classical"
    else:
        return "dfnet"

def classical(img, rect, p):
    avg_color = p.mean(axis=0).astype(np.uint8)
    img[rect[0]:rect[2], rect[1]:rect[3]] = avg_color

# In[]:
i = 10
j = 2

thresh = 0
histwidth = 5
neighborhood = 3

line  = content[i]

values = line.split(' ', 1)
img_id = values[0]
value = values[1]
val = json.loads(value) 

mask = 255 * np.ones((600,500), dtype=np.uint8)
img = cv2.imread(path_dataset+img_id+'.png')

rect = val[j]

p_gray = perimeter(img, rect, neighborhood = neighborhood, gray = True)
decision = classical_or_dfnet(p_gray, thresh = thresh, histwidth = histwidth)
print(decision)

if decision == "classical":
    p = perimeter(img, rect, neighborhood = 3, gray = False)
    classical(img, rect, p)

plt.imshow(img)

# In[]:




# In[]:




# In[]:







