import os
from multiprocessing import Pool
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-img-A-dir", default="data_dir/S2Looking/train/A")
    parser.add_argument("--input-img-B-dir", default="data_dir/S2Looking/train/B")
    parser.add_argument("--input-mask-dir",  default="data_dir/S2Looking/train/label")
    parser.add_argument("--output-txt-dir",  default="data_dir/S2Looking/train.txt")
    return parser.parse_args()

def process_file(filename):
    a_filename = filename
    b_filename = filename
    label_filename = filename.split('.')[0] + '.png'
    if not (b_filename in b_filenames and label_filename in label_filenames):
        return None
    # 构建每一组对应数据的绝对路径，并将路径添加到 result 列表中
    a_path = os.path.join(img_A_folder, filename)
    b_path = os.path.join(img_B_folder, b_filename)
    label_path = os.path.join(label_folder, label_filename)
    return f'{a_path}\t{b_path}\t{label_path}\t**\t**\n'

if __name__ == '__main__':
    args = parse_args()
    img_A_folder = args.input_img_A_dir
    img_B_folder = args.input_img_B_dir
    label_folder = args.input_mask_dir
    txt_dir = args.output_txt_dir

    a_filenames = os.listdir(img_A_folder)
    b_filenames = os.listdir(img_B_folder)
    label_filenames = os.listdir(label_folder)

    with Pool(16) as p:
        results = p.map(process_file, a_filenames)
    # Filter out None and write data_strs to txt file
    with open(txt_dir, 'w') as f:
        for result in results:
            if result is not None:
                f.write(result)