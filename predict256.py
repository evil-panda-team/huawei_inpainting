import numpy as np
import json
import cv2
import os

import json
import matplotlib.pyplot as plt
from collections import Counter

import tensorflow as tf
import neuralgym as ng
from tqdm import tqdm

import sys
sys.path.insert(1, './generative_inpainting/')

from inpaint_model import InpaintCAModel

def denoising():
    pass

def generative_inpainting(image, mask):
    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))
    model = InpaintCAModel()
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    checkpoint_dir = 'generative_inpainting/model_logs/release_imagenet_256/'
    
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
    tf.reset_default_graph()
    return result

def parse_masks_file(path_dataset, file_path, images_save_path, start_index=0):
    with open(file_path) as f:
        content = f.readlines()
        values = []
        i = start_index

        for line in content:
            img = cv2.imread(path_dataset+str(i)+'.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = 255 * np.zeros((600,500,3), dtype=np.uint8)
            result_img = 255 * np.ones((600,500,3), dtype=np.uint8)
 
            value = line.split(' ', 1)[1]
            val = json.loads(value)
            for rect in val:
                mask[rect[0]: rect[2], rect[1]:rect[3]] = 255
            
            img_part_1 = img[:256,:256,:]
            img_part_2 = img[:256,244:,:]
            img_part_3 = img[172:428,:256,:]
            img_part_4 = img[172:428,244:,:]
            img_part_5 = img[344:,:256,:]
            img_part_6 = img[344:,244:,:]

            mask_part_1 = mask[:256,:256]
            mask_part_2 = mask[:256,244:]
            mask_part_3 = mask[172:428,:256]
            mask_part_4 = mask[172:428,244:]
            mask_part_5 = mask[344:,:256]
            mask_part_6 = mask[344:,244:]

            result_1 = generative_inpainting(img_part_1, mask_part_1)
            result_2 = generative_inpainting(img_part_2, mask_part_2)
            result_3 = generative_inpainting(img_part_3, mask_part_3)
            result_4 = generative_inpainting(img_part_4, mask_part_4)
            result_5 = generative_inpainting(img_part_5, mask_part_5)
            result_6 = generative_inpainting(img_part_6, mask_part_6)

            result_img[:256,:256,:] = result_1
            result_img[:256,244:,:] = result_2
            result_img[172:428,:256,:] = result_3
            result_img[172:428,244:,:] = result_4
            result_img[344:,:256,:] = result_5
            result_img[344:,244:,:] = result_6

            img_blurred = result_img.copy()

            img_blurred = cv2.GaussianBlur(img_blurred, ksize=(17,13), sigmaX=4, sigmaY=7)
            img_blurred = cv2.medianBlur(img_blurred, 11)

            for rect in val:
                result_img[rect[0]: rect[2], rect[1]:rect[3]] = img_blurred[rect[0]: rect[2], rect[1]:rect[3]]

            cv2.imwrite('{}{}.png'.format(images_save_path, i), result_img)
            i += 1


if __name__ == '__main__':
    start_index = 800
    path_dataset = 'dataset/test/'
    images_save_path = 'output/gen_inpainting_imagenet/'
    os.makedirs(images_save_path, exist_ok=True)
    file_path = 'dataset/test/test_mask.txt'
    parse_masks_file(path_dataset, file_path,
                     images_save_path, start_index=start_index)
    # os.makedirs(test_imgs_save_path, exist_ok=True)
    # resize_test(path_dataset, test_imgs_save_path)
