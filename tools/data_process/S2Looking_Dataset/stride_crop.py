import cv2
import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="data_dir/S2Looking/train/A")
    parser.add_argument("--output-dir", default="data_dir/S2Looking/train/A_512_nooverlap")
    return parser.parse_args()


def slide_crop(filename):
    real_path = os.path.join(folder_path, filename)
    name = filename.split('.')[0]
    img = cv2.imread(real_path)
    crop_width, crop_height = 512, 512
    step_x, step_y = 512, 512
    height, width, _ = img.shape
    crops = []
    for i in range(0, width - crop_width + 1, step_x):
        for j in range(0, height - crop_height + 1, step_y):
            crop = img[j:j + crop_height, i:i + crop_width]
            crops.append(crop)
    for i, crop in enumerate(crops):
        cv2.imwrite(f'{out_path}/{name}_crop_{i}.png', crop)

if __name__ == '__main__':
    args = parse_args()
    folder_path = args.input_dir
    out_path = args.output_dir
    # os.system(f'sudo rm -rf {out_path}')
    os.system(f'mkdir {out_path}')

    filenames = [filename for filename in os.listdir(folder_path) if
                 filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif')]
    
    with Pool(16) as p:
        list(tqdm(p.imap(slide_crop, filenames), total=len(filenames)))
    
