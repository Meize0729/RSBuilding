import os
import shutil
from multiprocessing import Pool

# 文件复制函数
def copy_file(file_info):
    source_path, dest_dir = file_info
    shutil.copy(source_path, dest_dir)

# 处理每行数据的函数
def process_line(line, dirs):
    paths = line.strip().split('\t')
    if len(paths) >= 3:
        return [(paths[0], dirs[0]), (paths[1], dirs[1]), (paths[2], dirs[2])]
    return []

def main(txt_file_path, dir_im1, dir_im2, dir_label):
    # 确保目标目录存在，如果不存在则创建
    for dir_path in [dir_im1, dir_im2, dir_label]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    tasks = []

    # 读取文本文件并准备复制任务
    with open(txt_file_path, 'r') as file:
        for line in file:
            tasks.extend(process_line(line, [dir_im1, dir_im2, dir_label]))

    # 使用多进程进行文件复制
    with Pool(64) as pool:
        pool.map(copy_file, tasks)

    print("文件复制完成。")

# 假设的文本文件路径和目标目录路径
txt_file_path = '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/test_split.txt'
dir_im1 = '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/test/A'
dir_im2 = '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/test/B'
dir_label = '/mnt/public/usr/wangmingze/Datasets/CD/SECOND/test/label'

main(txt_file_path, dir_im1, dir_im2, dir_label)
