# -*- coding: utf-8 -*-

import os
from glob import glob
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import zipfile

import sys
sys.path.insert(1, './DFNet/')

from DFNet.test_vlad import Tester

# In[]:
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default = '../DATASET_INPAINTING/test/')
args = parser.parse_args()

# In[]:
#dataset_path = '../DATASET_INPAINTING/test_final/'
dataset_path = args.img_dir

# In[]:
def generate_mask(img, min_size=2000):
    thresh = cv2.inRange(img, (255,255,255), (255,255,255))
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    mask = 255*np.ones((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask[output == i + 1] = 0
        
    return mask

# In[]:
model = './DFNet/model/model_places2.pth'
input_size = 512
batch_size = 1
tester = Tester(model, input_size, batch_size, 'cpu')

zf = zipfile.ZipFile('result.zip', mode='w')

images = glob(dataset_path + '*.png')
images.sort()

for img_path in tqdm(images):
    
    k = img_path.split('/')[-1]
    
    mask = np.hstack((generate_mask(cv2.imread(img_path)), 255*np.ones((600,12), dtype=np.uint8)))
    
    # Upper part:
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[:512,:500]
    y_pred = tester.inpaint(img, mask[:512,:512])

    # Lower part
    img = np.zeros((512,512,3), dtype=np.uint8)
    img[:512,:500] = cv2.imread(img_path)[-512:,:500]
    y_pred2 = tester.inpaint(img, mask[-512:,:512])
    
    # Concatenate
    img = cv2.imread(img_path)
    img[:512,:500] = y_pred[:512,:500]
    img[512:,:500] = y_pred2[-(600-512):,:500]
    
    # BLURRING
    img_blurred = img.copy()
    
    #Postprocessing
    img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, 1)
    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
    img_blurred = cv2.medianBlur(img_blurred, 11)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i,j] == 0:
                img[i,j] = img_blurred[i,j]
                
    cv2.imwrite("{}".format(k), img)
    zf.write(k)
    os.remove(k)
    
zf.close()