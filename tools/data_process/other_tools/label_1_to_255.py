import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import tifffile


def process_image(filepath):
    # 读取图像
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img[img != 0] = 255
    os.system(f'rm -rf {filepath}')
    # 保存调整后的图像
    cv2.imwrite(filepath, img)

if __name__ == '__main__':
    # 文件夹路径

    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON/train.txt'     
    image_files = []
    with open(folder_path, 'r') as f:
        for line in f.readlines():
            a, b, cd, label_a, label_b = line.strip().split('\t')
            image_files.append(cd)

        # 使用多进程并行处理图像
        with Pool(80) as p:
            list(tqdm(p.imap(process_image, image_files), total=len(image_files)))