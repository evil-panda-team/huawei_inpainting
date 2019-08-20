# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

#import imgaug.augmenters as iaa
#from albumentations import (
#    OneOf,
#    Blur,
#    RandomGamma,
#    HueSaturationValue,
#    RGBShift,
#    RandomBrightness,
#    RandomContrast,
#    MedianBlur,
#    CLAHE
#)

# In[]: Augmentations
def random_float(low, high):
    return np.random.random()*(high-low) + low

def augment(image):
    
    mul = random_float(0.1, 0.5)
    add = np.random.randint(-100,-50)
    gamma = random_float(2,3)
    
    aug = iaa.OneOf([
            iaa.Multiply(mul = mul),
            iaa.Add(value = add),
            iaa.GammaContrast(gamma=gamma)
            ])

    image_augmented = aug.augment_image(image)
    
    return image_augmented

def augment_hard(image):
    
    aug = OneOf([
        Blur(blur_limit=5, p=1.),
        RandomGamma(gamma_limit=(50, 150), p=1.),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
        RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
        RandomBrightness(limit=.25, p=1.),
        RandomContrast(limit=.25, p=1.),
        MedianBlur(blur_limit=5, p=1.),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
        ], p=1.)

    augmented = aug(image=image)
    image_augmented = augmented['image']
    
    return image_augmented

# In[]:
def psnr(x, y):
   mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
   if mse == 0:
       return 100
   return 20 * math.log10(255.0 / math.sqrt(mse))

def l1_loss(x, y):
   return np.sum(np.abs(x / 256. - y / 256.)) / 3.

# In[]:
train_path_dataset = '../DATASET_INPAINTING/gt/'
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
images_pred = np.zeros((len(dic),600,500,3), dtype=np.uint8)
images_gt = np.zeros((len(dic),600,500,3), dtype=np.uint8)

i = 0
for i,(k,v) in tqdm(enumerate(dic.items())):
    img_path = path_dataset + k
    image_gt = cv2.imread(img_path)
    images_gt[i] = image_gt
    pred_path = preds_path + k
    image_pred = cv2.imread(pred_path)
    images_pred[i] = image_pred
    i += 1
    
print("IMAGES READING COMPLETED")

# In[]:
#bbox_w_max = 0
#bbox_h_max = 0
#area_max = 0
#bbox_w_min = 500
#bbox_h_min = 600
#area_min = math.inf
#
#for i,(k,v) in enumerate(dic.items()):
##    if k != '800.png':
##        continue
#    for bbox in v:
#        if bbox[3] > 499:
#            bbox[3] = 499
#        bbox_w = bbox[2]-bbox[0]
#        bbox_h = bbox[3]-bbox[1]
#        area = bbox_w*bbox_h
#        if bbox_w < bbox_w_min:
#            bbox_w_min = bbox_w
#        if bbox_w > bbox_w_max:
#            bbox_w_max = bbox_w
#        if bbox_h < bbox_h_min:
#            bbox_h_min = bbox_h
#        if bbox_h > bbox_h_max:
#            bbox_h_max = bbox_h
#        if area > area_max:
#            area_max = area
#        if area < area_min:
#            area_min = area
        
#avg_psnr = 0
#
#for i in range(len(dic)):
#    image_gt = images_gt[i]
#    image_pred = images_pred[i]
#    score = psnr(image_gt, image_pred)
#    avg_psnr = avg_psnr + score/len(dic)
#    
#print("AVERAGE PSNR: {}".format(avg_psnr))
#AVERAGE PSNR: 30.38167518624206

# In[]: 30.3816
avg_psnr = 0

ksize = 1

for i,(k,v) in tqdm(enumerate(dic.items())):
    image_gt = images_gt[i].copy()
    image_pred = images_pred[i].copy()
    
    img_blurred = image_pred.copy()
   
    img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, ksize)
    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
    img_blurred = cv2.medianBlur(img_blurred, 11)

    for bbox in v:
        image_pred[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    
    score = psnr(image_gt, image_pred)
    avg_psnr = avg_psnr + score/len(dic)

print("AVERAGE PSNR BLURRED: {} with Kernel size = {}".format(avg_psnr, ksize))
    
# In[]:
#for ksize in range(1,45,2):
#    avg_psnr = 0
#    for i,(k,v) in enumerate(dic.items()):
#        image_gt = images_gt[i].copy()
#        image_pred = images_pred[i].copy()
#        
#        img_blurred = image_pred.copy()
#        img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, ksize)
#        img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
#        img_blurred = cv2.medianBlur(img_blurred, 11)
#        
#        for bbox in v:
#            image_pred[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]
#        
#        score = psnr(image_gt, image_pred)
#        avg_psnr = avg_psnr + score/len(dic)
#    
#    print("AVERAGE PSNR BLURRED: {} with Kernel size = {}".format(avg_psnr, ksize))    

# In[]:
#AVERAGE PSNR BLURRED: 31.45794571502578
#img_blurred = cv2.medianBlur(image_pred, 9)
#img_blurred = cv2.GaussianBlur(img_blurred, (9, 9), 0)
#img_blurred = cv2.blur(img_blurred, (13, 13))
#img_blurred = cv2.bilateralFilter(img_blurred, 9, 75, 75)

# Median blur only: cv2.medianBlur(img_blurred, ksize)
#AVERAGE PSNR BLURRED: 31.173959681956486 with Kernel size = 7
#AVERAGE PSNR BLURRED: 31.227299282543832 with Kernel size = 9
#AVERAGE PSNR BLURRED: 31.258322249913054 with Kernel size = 11
#AVERAGE PSNR BLURRED: 31.272085221856834 with Kernel size = 13
#AVERAGE PSNR BLURRED: 31.274012976724027 with Kernel size = 15
#AVERAGE PSNR BLURRED: 30.892767156228217 with Kernel size = 17

# Gaussian blur only: cv2.GaussianBlur(img_blurred, (ksize, ksize), 0)
#AVERAGE PSNR BLURRED: 31.185261664285637 with Kernel size = 7
#AVERAGE PSNR BLURRED: 31.343135152288475 with Kernel size = 9
#AVERAGE PSNR BLURRED: 31.15105996958201 with Kernel size = 11
#AVERAGE PSNR BLURRED: 31.424063511197307 with Kernel size = 13
#AVERAGE PSNR BLURRED: 31.206647308801024 with Kernel size = 15
#AVERAGE PSNR BLURRED: 31.473614987938376 with Kernel size = 17
#AVERAGE PSNR BLURRED: 31.377409866079596 with Kernel size = 19
#AVERAGE PSNR BLURRED: 31.23852756078255 with Kernel size = 21
    
# Averaging (Blur) only: cv2.blur(img_blurred, (ksize, ksize))
#AVERAGE PSNR BLURRED: 31.28724511667665 with Kernel size = 7
#AVERAGE PSNR BLURRED: 31.333292908443095 with Kernel size = 9
#AVERAGE PSNR BLURRED: 31.353393412458612 with Kernel size = 11
#AVERAGE PSNR BLURRED: 31.35540737675998 with Kernel size = 13
#AVERAGE PSNR BLURRED: 31.346087596784898 with Kernel size = 15
#AVERAGE PSNR BLURRED: 31.329656699581278 with Kernel size = 17
#AVERAGE PSNR BLURRED: 31.308369019091298 with Kernel size = 19
#AVERAGE PSNR BLURRED: 31.283616107775497 with Kernel size = 21

# In bilateral filter only: cv2.bilateralFilter(img_blurred, ksize, 75, 75)
#AVERAGE PSNR BLURRED: 30.886578148761966 with Kernel size = 7
#AVERAGE PSNR BLURRED: 30.936907245548614 with Kernel size = 9
#AVERAGE PSNR BLURRED: 30.979952853944216 with Kernel size = 11
#AVERAGE PSNR BLURRED: 31.00344631367733 with Kernel size = 13
#AVERAGE PSNR BLURRED: 31.021308961974626 with Kernel size = 15
#AVERAGE PSNR BLURRED: 31.036538780291252 with Kernel size = 17
#AVERAGE PSNR BLURRED: 31.04742717068983 with Kernel size = 19
#AVERAGE PSNR BLURRED: 31.054368528493377 with Kernel size = 21
#AVERAGE PSNR BLURRED: 31.057914792376575 with Kernel size = 23
#AVERAGE PSNR BLURRED: 31.059490859534854 with Kernel size = 25
#AVERAGE PSNR BLURRED: 31.060809335704224 with Kernel size = 27
#AVERAGE PSNR BLURRED: 31.06114205567786 with Kernel size = 29
#AVERAGE PSNR BLURRED: 31.06028618644941 with Kernel size = 31
#AVERAGE PSNR BLURRED: 31.059345160222747 with Kernel size = 33

#######################################################################
    
# GAUSIIAN BLUR BRUTFORCE
#cv2.GaussianBlur(img_blurred, ksize=(17,17), sigmaX=4, sigmaY=7)   31.500192703254925
#cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)   31.518446587802206
    
# GAUSIIAN BLUR + MEDIAN BLUR
# 31.522442211210016
#cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7) 
#cv2.medianBlur(img_blurred, 11)
    
# img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, 43) + Previous
# AVERAGE PSNR BLURRED: 31.54025696026921

# 31.584629180915538
#    img_path = dataset_path + k
#    
#    pad = False
#    for bbox in v:
#        bb3 = bbox[3]
#        if bb3 >= 500 or bbox[0]==0 or bbox[1]==0:
#            pad = True
#        
#    if pad:
#        mask = 255*np.ones((600,500), dtype=np.uint8)
#        for bbox in v:
#            bb3 = bbox[3]
#            mask[bbox[0]:bbox[2], bbox[1]:bb3] = 0
#        
#        mask = cv2.copyMakeBorder(mask,3,0,6,6,cv2.BORDER_REPLICATE)
#        
#        pad_right = cv2.copyMakeBorder(cv2.imread(img_path),3,0,6,6,cv2.BORDER_REPLICATE)
#        img = pad_right[:512,:512].copy()
#    else:
#        mask = 255*np.ones((600,512), dtype=np.uint8)
#        for bbox in v:
#            bb3 = bbox[3]
#            mask[bbox[0]:bbox[2], bbox[1]:bb3] = 0
#        img = np.zeros((512,512,3), dtype=np.uint8)
#        img[:512,:500] = cv2.imread(img_path)[:512,:500]
#        
#    y_pred = tester.inpaint(img, mask[:512,:512])
#    
#    # Concatenate
#    img = cv2.imread(img_path)
#    if pad:
#        img[:500,:500] = y_pred[3:503,6:506]
#    else:
#        img[:512,:500] = y_pred[:512,:500]
#    
#    # Postprocessing
#    img_blurred = img.copy()
##    img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred, None, 4, 4, 39, 39)
#    img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
#    img_blurred = cv2.medianBlur(img_blurred, 11)
#
#    for bbox in v:
#        img[bbox[0]:bbox[2],bbox[1]:bbox[3]] = img_blurred[bbox[0]:bbox[2],bbox[1]:bbox[3]]