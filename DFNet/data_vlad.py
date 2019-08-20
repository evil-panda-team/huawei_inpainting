import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# In[]:            
class DS(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        with open(root) as f:
            content = f.readlines()
        for line in content:
            self.samples.append(line.rstrip()) #rstrip to remove newline characters
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)
            
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = Image.open(sample_path).convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        mask = DS.random_mask()
        mask = torch.from_numpy(mask)

        return sample, mask
    
    @staticmethod
    def random_mask(height=512, width=512, bbox_side_min = 80, bbox_side_max = 127):

        mask = np.ones((height, width))
        
        bboxes_num = np.random.randint(1,5)
        
        for i in range(bboxes_num):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            w = np.random.randint(bbox_side_min, bbox_side_max)
            h = np.random.randint(bbox_side_min, bbox_side_max)
            cv2.rectangle(mask, (x-bbox_side_min, y-bbox_side_min), (x-bbox_side_min+w, y-bbox_side_min+h), 0., -1)
            
        if np.random.random() < 0.5:
            mask = np.fliplr(mask)
        if np.random.random() < 0.5:
            mask = np.flipud(mask)
            
        return mask.reshape((1,)+mask.shape).astype(np.float32) 