import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

# 这是你的字符串列表
# strings = ['fc_siam_conc', 'fc_siam_diff', 'bit_r18', 'snunet_c32', 'tinycd', 'tinycd_v2_l', 
#            'changeformer_mitb0', 'changeformer_mitb1', 'changer_s50', 'ViT_L_finetune']

strings_cd = [
    '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_CD/test/label_cd',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/diff/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/bit/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/tinycd/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/changeformer_mitb0/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/changer_s50/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/vitl/compare_pixel'
]
names_cd = ['label.png', 'diff.png', 'bit.png', 'tinycd.png', 'changeformer.png', 'changer.png', 'vitl.png']
names_bx = ['img.png', 'label.png', 'unet.png', 'deeplabv3.png', 'segformer.png', 'buildformer.png', 'pred.png']
strings_bx = [
    '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_BX/test/AB',
    '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_BX/test/label',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/unet/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/deeplabv3/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/segformer/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/buildformer/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/vitl/compare_pixel'
]

names = ['bj_t1VSt2_L81_00655_784311_crop_8.png', 'sh_t1VSt2_L81_892807_780503_crop_3.png', 'bj_t1VSt2_L81_00655_784111_crop_5.png', 'bj_t1VSt3_L81_00839_784255_crop_4.png']

for file_name in names:
    place = file_name.split('_')[0]
    time1 = file_name.split('VS')[0].split('_')[-1]
    time2 = file_name.split('VS')[1].split('_')[0]
    pic = file_name.split('VS')[1].replace(f'{time2}_', '')
    dir_name = '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bandon/' + pic.replace('.png', '')
    os.system(f'sudo rm -rf {dir_name}')
    os.system(f'sudo mkdir {dir_name}')
    for i, string in enumerate(strings_cd):
        img_path = os.path.join(string, file_name)
        pic_name = os.path.join(dir_name, names_cd[i])
        os.system(f'sudo cp {img_path} {pic_name}')

    img_name_1 = place + '_' + time1 + '_' + pic
    img_name_2 = place + '_' + time2 + '_' + pic
    for i, string in enumerate(strings_bx):
        img_path_1 = os.path.join(string, img_name_1)
        # import pdb; pdb.set_trace()
        if not os.path.exists(img_path_1):
            img_path_1 = img_path_1.replace('.png', '.jpg')
        img_new_name_1 = os.path.join(dir_name, time1 + names_bx[i])
        os.system(f'sudo cp {img_path_1} {img_new_name_1}')
        img_path_2 = os.path.join(string, img_name_2)
        if not os.path.exists(img_path_2):
            img_path_2 = img_path_2.replace('.png', '.jpg')
        img_new_name_2 = os.path.join(dir_name, time2 + names_bx[i])
        os.system(f'sudo cp {img_path_2} {img_new_name_2}')