import numpy as np
import json
import cv2
import os
from tqdm import tqdm
from skimage.restoration import denoise_wavelet, cycle_spin


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

            img_blurred = img.copy()

            # img_blurred = cycle_spin(np.float32(img_blurred / 255), func=denoise_wavelet, max_shifts=9,
            #                          func_kw=denoise_kwargs, multichannel=True)
            img_blurred = np.uint8(img_blurred * 255)
            # img_blurred = cv2.bilateralFilter(img_blurred,9,75,75)
            # img_blurred = denoising(img_blurred)
            # img_blurred = cv2.fastNlMeansDenoisingColored(img_blurred,None,10,10,7,21)
            img_blurred = cv2.GaussianBlur(
                img_blurred, ksize=(17, 13), sigmaX=4, sigmaY=7)
            img_blurred = cv2.medianBlur(img_blurred, 11)
            value = line.split(' ', 1)[1]
            val = json.loads(value)
            for rect in val:
                # Best
                img[rect[0]: rect[2], rect[1]:rect[3]
                    ] = img_blurred[rect[0]: rect[2], rect[1]:rect[3]]

            cv2.imwrite('{}{}.png'.format(images_save_path, i), img)
            i += 1


if __name__ == '__main__':
    start_index = 800
    path_dataset = 'output/dfnet/result/'
    images_save_path = 'output/dfnet/result_postprocess/'
    os.makedirs(images_save_path, exist_ok=True)
    file_path = 'dataset/test/test_mask.txt'

    parse_masks_file(path_dataset, file_path,
                     images_save_path, start_index=start_index)
    # os.makedirs(test_imgs_save_path, exist_ok=True)
    # resize_test(path_dataset, test_imgs_save_path)
