# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default="/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_tmp")
    parser.add_argument("--output-dir", default="/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_tmp.txt")
    parser.add_argument("--split",      default="test")
    return parser.parse_args()


def generate_txt_from_dir(src_dir, dst_dir, split):
    """Generate .txt file for BANDON dataset.

    Args:
        src_dir (str): path of the source dataset.
        dst_dir (str): Path to save .txt file.
        split (str): sub_dirs. 'train', 'val' or 'test'

    """

    label_subdir_name = 'labels_unch0ch1ig255' if split == 'train' or split == 'test' \
        else 'labels'
    
    file_list = []
    label_subdir = osp.join(src_dir, label_subdir_name)
    for city in os.listdir(label_subdir):
        city_subdir = osp.join(label_subdir, city)
        for time in os.listdir(city_subdir):
            time_subdir = osp.join(city_subdir, time)
            for img_name in sorted(os.listdir(time_subdir)):
                t1, t2 = time.split('VS')
                img_name_ = osp.splitext(img_name)[0]
                file_list.append([
                    osp.join(src_dir, 'imgs', city, t1, img_name_ + '.jpg'),
                    osp.join(src_dir, 'imgs', city, t2, img_name_ + '.jpg'),
                    osp.join(src_dir, label_subdir_name, city, time, img_name_ + '.png'),
                    osp.join(src_dir, 'building_labels', city, t1, img_name_ + '.png'),
                    osp.join(src_dir, 'building_labels', city, t2, img_name_ + '.png'),
                ])
    with open(dst_dir, 'w') as f:
        for item in file_list:
            f.write('\t'.join(item) + '\n')


def main():
    args = parse_args()
    dataset_path = args.input_dir
    out_dir = args.output_dir
    split = args.split
    # dataset_path = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON'
    # out_dir = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON'

    print('Making .txt files ...')
    generate_txt_from_dir(dataset_path, out_dir, split)
    print('Done!')


if __name__ == '__main__':
    main()


