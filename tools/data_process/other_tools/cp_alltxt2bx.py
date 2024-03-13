import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm
# 这是你想要复制到的两个路径
dst_path_ab = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_BX/train/AB'
dst_path_label = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_BX/train/label'
os.system(f'sudo rm -rf {dst_path_ab}')
os.system(f'sudo rm -rf {dst_path_label}')

# 创建目标路径
os.makedirs(dst_path_ab, exist_ok=True)
os.makedirs(dst_path_label, exist_ok=True)

def copy_file(args):
    src, dst = args
    shutil.copy(src, dst)

with open('/mnt/public/usr/wangmingze/Datasets/CD/BANDON/train_bx.txt', 'r') as fr:
    lines = fr.readlines()

# 创建一个进程池
with Pool() as p:
    tasks = []
    for line in lines:
        a, _, _, label_a, _ = line.strip().split('\t')

        # 获取文件名和路径的各个部分
        parts_a = a.split('/')
        parts_label_a = label_a.split('/')

        # 组合新的文件名
        file_name_a = parts_a[-3] + '_' + parts_a[-2] + '_' + parts_a[-1].rsplit('.', 1)[0] + '.' + parts_a[-1].rsplit('.', 1)[1]
        file_name_label_a = parts_label_a[-3] + '_' + parts_label_a[-2] + '_' + parts_label_a[-1].rsplit('.', 1)[0] + '.' + parts_label_a[-1].rsplit('.', 1)[1]

        # 添加任务
        tasks.append((a, os.path.join(dst_path_ab, file_name_a)))
        tasks.append((label_a, os.path.join(dst_path_label, file_name_label_a)))

    # 使用tqdm显示进度
    for _ in tqdm(p.imap_unordered(copy_file, tasks), total=len(tasks)):
        pass