import os
import cv2
import numpy as np
import mmcv
import mmengine.fileio as fileio
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter

def process_image(image_file):
    # image_path = os.path.join(folder_path, image_file)
    img_bytes_from = fileio.get(
        image_file, backend_args=None)
    image = mmcv.imfrombytes(
        img_bytes_from, flag='unchanged',
        backend=None).squeeze().astype(np.uint8)
    # gt_semantic_seg_from_copy = image.copy()
    # image[gt_semantic_seg_from_copy < 128] = 0
    # image[gt_semantic_seg_from_copy >= 128] = 1
    image_values = image.flatten()
    return Counter(image_values)

def read_images_from_folder(folder_path):
    image_files = []
    with open(folder_path, 'r') as f:
        for line in f.readlines():
            a, b, cd, label_a, label_b = line.strip().split('\t')
            image_files.append(cd)
    with Pool(os.cpu_count()) as p:
        all_values_list = list(tqdm(p.imap(process_image, image_files), total=len(image_files)))
    all_values = sum(all_values_list, Counter())
    return all_values

folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_cd.txt'  # 替换为你的文件夹路径
all_values = read_images_from_folder(folder_path)
print(all_values)

# def process_image(image_file):
#     image_path = os.path.join(folder_path, image_file)
#     img_bytes_from = fileio.get(
#         image_path, backend_args=None)
#     image = mmcv.imfrombytes(
#         img_bytes_from, flag='unchanged',
#         backend=None).astype(np.uint8)
#     image_values = image.flatten()
#     import pdb; pdb.set_trace()
#     return Counter(image_values)

# def read_images_from_folder(folder_path):
#     image_files = os.listdir(folder_path)
#     all_values = Counter()
#     for image_file in tqdm(image_files):
#         all_values += process_image(image_file)
#     return all_values

# folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/WHU-main/train/label'  # 替换为你的文件夹路径
# all_values = read_images_from_folder(folder_path)
# print(all_values)