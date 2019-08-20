import numpy as np
import json
import cv2
import os


def parse_masks_file(path_dataset, file_path, masks_save_path, images_save_path, start_index=0):
    with open(file_path) as f:
        content = f.readlines()
        values = []
        i = start_index
        for line in content:
            mask = 255 * np.ones((512, 512), dtype=np.uint8)
            img = cv2.imread(path_dataset+str(i)+'.png')
            # cv2.line(img, (0, 512), (500, 512), (0, 0, 255), 1)
            # img_resized = cv2.resize(img, (512, 512))
            img_resized = np.zeros((512, 512, 3))
            img_resized[0:512, 0:500, :] = img[0:512, 0:500, :]
            cv2.imwrite('{}{}.png'.format(images_save_path, i), img_resized)
            value = line.split(' ', 1)[1]
            val = json.loads(value)
            for rect in val:
                # mask[int((512/600)*rect[0]):int((512/600)+1
                #                                 * rect[2]), int((512/500)*rect[1]):int((512/500)*rect[3])+1] = 0
                mask[rect[0]: rect[2], rect[1]:rect[3]]=0
            # mask_resized = cv2.resize(mask, (512, 512))
            cv2.imwrite('{}{}.png'.format(masks_save_path, i), mask)
            i += 1


def resize_test(test_images_path, save_path):
    for root, dirnames, filenames in os.walk(test_images_path):
        for f in filenames:
            if f.endswith('.png'):
                img=cv2.imread(os.path.join(root, f))
                img_resized=cv2.resize(img, (512, 512))
                cv2.imwrite(os.path.join(save_path, f), img_resized)


if __name__ == '__main__':
    mode='test'
    if mode == 'train':
        start_index=0
    else:
        start_index=800
    path_dataset='dataset/{}/'.format(mode)
    test_imgs_save_path='dfnet_dataset_test/'
    masks_save_path='dataset/dfnet/{}/masks/'.format(mode)
    images_save_path='dataset/dfnet/{}/images/'.format(mode)
    os.makedirs(masks_save_path, exist_ok = True)
    os.makedirs(images_save_path, exist_ok = True)
    file_path=path_dataset + '{}_mask.txt'.format(mode)
    parse_masks_file(path_dataset, file_path,
                     masks_save_path, images_save_path, start_index = start_index)
    # os.makedirs(test_imgs_save_path, exist_ok=True)
    # resize_test(path_dataset, test_imgs_save_path)
