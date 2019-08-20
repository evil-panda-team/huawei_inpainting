# -*- coding: utf-8 -*-

import os
from random import shuffle
from tqdm import tqdm

# In[]:
folder_path = '../../../colddata/datasets/places/'
train_filename = 'train_shuffled.flist'
is_shuffled = '1'

# In[]:
training_file_names = list()
for (dirpath, dirnames, filenames) in tqdm(os.walk(folder_path + "train_large")):
    training_file_names += [os.path.join(dirpath, file) for file in filenames]
    
# shuffle file names if set
if is_shuffled == 1:
    shuffle(training_file_names)

# make output file if not existed
if not os.path.exists(train_filename):
    os.mknod(train_filename)

fo = open(train_filename, "w")
fo.write("\n".join(training_file_names))
fo.close()

# print process
print("Written file is: ", train_filename, ", is_shuffle: ", is_shuffled)