import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from skimage.restoration import inpaint_biharmonic


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def parse_masks_file(path_dataset, file_path, images_save_path, start_index=0):
    with open(file_path) as f:
        content = f.readlines()
        values = []
        i = start_index

        denoise_kwargs = dict(
            multichannel=True, convert2ycbcr=True, wavelet='db1')

        for line in tqdm(content):
            img = cv2.imread(path_dataset+str(i)+'.png')
            mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)

            value = line.split(' ', 1)[1]
            val = json.loads(value)
            for rect in val:
                # Best
                mask[rect[0]: rect[2], rect[1]:rect[3]] = 255
            out = cv2.inpaint(img,mask,45,cv2.INPAINT_TELEA)
            cv2.imwrite('{}{}.png'.format(images_save_path, i), out)
            i += 1


if __name__ == '__main__':
    start_index = 800
    path_dataset = 'output/dfnet/result/'
    images_save_path = 'output/dfnet/result_classical_biharmonic/'
    os.makedirs(images_save_path, exist_ok=True)
    file_path = 'dataset/test/test_mask.txt'

    parse_masks_file(path_dataset, file_path,
                     images_save_path, start_index=start_index)
    # os.makedirs(test_imgs_save_path, exist_ok=True)
    # resize_test(path_dataset, test_imgs_save_path)
