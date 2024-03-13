import os
import re
from multiprocessing import Pool

pattern = re.compile(r'\d+')
def process_file(filename):
    match = pattern.search(filename)
    a_filename = filename[:match.end()] + '_pre_disaster' + filename[match.end():]
    b_filename = filename[:match.end()] + '_post_disaster' + filename[match.end():]
    # 构建对应的 B 和标签文件名
    # a_filename = filename.split('.')[0] + '_pre_disaster' + '.png'
    # b_filename = filename.split('.')[0] + '_post_disaster' + '.png'
    if not (a_filename in ab_filenames and b_filename in ab_filenames):
        return None
    # 构建每一组对应数据的绝对路径，并将路径添加到 result 列表中
    a_image = os.path.join(image_folder, a_filename)
    b_image = os.path.join(image_folder, b_filename)
    a_path = os.path.join(ab_folder, a_filename)
    b_path = os.path.join(ab_folder, b_filename)
    label_path = os.path.join(label_folder, filename)
    return f'{a_image}\t{b_image}\t{label_path}\t{a_path}\t{b_path}\n'


# def process_file(filename):
#     # 构建对应的 B 和标签文件名
#     a_filename = filename.split('.')[0] + '_pre_disaster' + '.png'
#     b_filename = filename.split('.')[0] + '_post_disaster' + '.png'
#     if not (a_filename in ab_filenames and b_filename in ab_filenames):
#         return None
#     # 构建每一组对应数据的绝对路径，并将路径添加到 result 列表中
#     a_image = os.path.join(image_folder, a_filename)
#     b_image = os.path.join(image_folder, b_filename)
#     a_path = os.path.join(ab_folder, a_filename)
#     b_path = os.path.join(ab_folder, b_filename)
#     label_path = os.path.join(label_folder, filename)
#     return f'{a_image}\t{b_image}\t{label_path}\t{a_path}\t{b_path}\n'

# 定义 A、B 和标签文件夹的路径
folder = '/mnt/public/usr/wangmingze/Datasets/CD/xView2_test'
mode = folder.split('/')[-1]
image_folder = folder + '/images_512_nooverlap'
ab_folder = folder + '/AB_512_nooverlap'
label_folder = folder + '/CD_512_nooverlap'

# 列出 A，B和label 文件夹下的所有文件名
ab_filenames = os.listdir(ab_folder)
cd_filenames = os.listdir(label_folder)
# 使用 Pool 对象创建多个进程
with Pool(10) as p:
    results = p.map(process_file, cd_filenames)
# Filter out None and write data_strs to txt file
with open(f'{folder}/test_nooverlap.txt', 'w') as f:
    for result in results:
        if result is not None:
            f.write(result)