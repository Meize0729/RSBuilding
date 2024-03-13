'''
用于把一个bandon txt cp成cd额数据
'''
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

# 这是你想要复制到的三个路径
dst_path_a = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_CD/test/A'
dst_path_b = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_CD/test/B'
dst_path_cd = '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_CD/test/label_cd'

os.system(f'sudo rm -rf {dst_path_a}')
os.system(f'sudo rm -rf {dst_path_b}')
os.system(f'sudo rm -rf {dst_path_cd}')

# 创建目标路径
os.makedirs(dst_path_a, exist_ok=True)
os.makedirs(dst_path_b, exist_ok=True)
os.makedirs(dst_path_cd, exist_ok=True)

def copy_file(args):
    src, dst = args
    shutil.copy(src, dst)

with open('/mnt/public/usr/wangmingze/Datasets/CD/BANDON/test_nooverlap.txt', 'r') as fr:
    lines = fr.readlines()

# 创建一个进程池
with Pool(64) as p:
    tasks = []
    for line in lines:
        a, b, cd, label_a, label_b = line.strip().split('\t')

        # 获取文件名和路径的各个部分
        parts_a = a.split('/')
        parts_b = b.split('/')
        parts_label_a = cd.split('/')

        # 组合新的文件名
        #             sh                  t2VSt3                    name                            
        file_name_a = parts_a[-3] + '_' + parts_label_a[-2] + '_' + parts_a[-1].rsplit('.', 1)[0] + '.' + parts_a[-1].rsplit('.', 1)[1]
        file_name_b = parts_b[-3] + '_' + parts_label_a[-2] + '_' + parts_b[-1].rsplit('.', 1)[0] + '.' + parts_b[-1].rsplit('.', 1)[1]
        file_name_label_a = parts_label_a[-3] + '_' + parts_label_a[-2] + '_' + parts_label_a[-1].rsplit('.', 1)[0] + '.' + parts_label_a[-1].rsplit('.', 1)[1]

        # 添加任务
        tasks.append((a, os.path.join(dst_path_a, file_name_a)))
        tasks.append((b, os.path.join(dst_path_b, file_name_b)))
        tasks.append((cd, os.path.join(dst_path_cd, file_name_label_a)))

    # 使用tqdm显示进度
    for _ in tqdm(p.imap_unordered(copy_file, tasks), total=len(tasks)):
        pass