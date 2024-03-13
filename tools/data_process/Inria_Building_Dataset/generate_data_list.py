import os
from multiprocessing import Pool
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-img-dir",  default="/mnt/public/usr/wangmingze/Datasets/CD/tmp/inria/AerialImageDataset/train/images")
    parser.add_argument("--input-mask-dir", default="/mnt/public/usr/wangmingze/Datasets/CD/tmp/inria/AerialImageDataset/train/gt")
    parser.add_argument("--output-txt-dir", default="/mnt/public/usr/wangmingze/Datasets/CD/tmp/inria/AerialImageDataset/val/images_512_nooverlap")
    return parser.parse_args()

def process_file(filename):
    # 构建对应的 B 和标签文件名
    pic_filename = filename
    label_filename = filename.split('.')[0] + '.png'
    if not (label_filename in label_filenames):
        return None
    a_path = os.path.join(img_folder, pic_filename)
    label_path = os.path.join(label_folder, label_filename)
    return f'{a_path}\t**\t**\t{label_path}\t**\n'

if __name__ == '__main__':
    args = parse_args()
    img_folder = args.input_img_dir
    label_folder = args.input_mask_dir
    txt_dir = args.output_txt_dir

    a_filenames = os.listdir(img_folder)
    label_filenames = os.listdir(label_folder)

    with Pool(16) as p:
        results = p.map(process_file, a_filenames)
    # Filter out None and write data_strs to txt file
    with open(txt_dir, 'w') as f:
        for result in results:
            if result is not None:
                f.write(result)