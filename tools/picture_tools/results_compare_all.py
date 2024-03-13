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
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/diff/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/bit/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/tinycd/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/changeformer_mitb0/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/changer_s50/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/vitl/compare_pixel'
]

strings_bx = [
    '/mnt/public/usr/wangmingze/Datasets/CD/BANDON_BX/test/AB',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/unet/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/deeplabv3/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/segformer/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/buildformer/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/bx/vitl/compare_pixel'
]

# 创建保存路径
save_dir = '/mnt/public/usr/wangmingze/opencd/pictures_tmp/tmp_bandon2'
os.makedirs(save_dir, exist_ok=True)

# 获取第一个任务路径下的所有文件名
file_names_cd = os.listdir(os.path.join(strings_cd[0]))
file_names_cd = ['sh_t1VSt2_L81_892807_780503_crop_3.png', 'bj_t2VSt3_L81_00687_784271_crop_2.png', 'bj_t1VSt3_L81_00839_784255_crop_4.png', 'bj_t1VSt2_L81_00687_784271_crop_10.png', 'bj_t1VSt2_L81_00655_784111_crop_5.png', 'bj_t1VSt2_L81_00655_784199_crop_1.png', 'bj_t1VSt2_L81_00655_784311_crop_8.png']
def process_file(file_name):
    # 提取时序名字
    place = file_name.split('_')[0]
    time1 = file_name.split('VS')[0].split('_')[-1]
    time2 = file_name.split('VS')[1].split('_')[0]
    pic = file_name.split('VS')[1].replace(f'{time2}_', '')
    img_name_1 = place + '_' + time1 + '_' + pic
    img_name_2 = place + '_' + time2 + '_' + pic

    img_path = os.path.join('/mnt/public/usr/wangmingze/opencd/pictures_for_all_ablation/cd/vitl/compare_pixel', file_name)
    img = mpimg.imread(img_path)
    nonzero_pixels = np.count_nonzero(img)
    total_pixels = img.shape[0] * img.shape[1]
    nonzero_ratio = nonzero_pixels / total_pixels
    if nonzero_ratio < 0.10:
        return None

    # 创建一个新的图像，3行，最大行数为6
    fig, axs = plt.subplots(3, 6, figsize=(40, 24))

    
    # 第一行：变化检测任务
    for i, string in enumerate(strings_cd):
        img_path = os.path.join(string, file_name)
        img = mpimg.imread(img_path)
        axs[0, i].imshow(img)
        axs[0, i].set_title(string.split('/')[-3])
        axs[0, i].axis('off')

    # 第二、三行：建筑提取任务
    for i, string in enumerate(strings_bx):
        # import pdb; pdb.set_trace()
        img_path_1 = os.path.join(string, img_name_1)
        if not os.path.exists(img_path_1):
            img_path_1 = img_path_1.replace('.png', '.jpg')
        img_1 = mpimg.imread(img_path_1)

        img_path_2 = os.path.join(string, img_name_2)
        if not os.path.exists(img_path_2):
            img_path_2 = img_path_2.replace('.png', '.jpg')
        img_2 = mpimg.imread(img_path_2)

        axs[1, i].imshow(img_1)
        axs[1, i].set_title(string.split('/')[-3])
        axs[1, i].axis('off')

        axs[2, i].imshow(img_2)
        axs[2, i].set_title(string.split('/')[-3])
        axs[2, i].axis('off')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 保存图片
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# for filename in file_names_cd:
#     process_file(filename)

# import pdb; pdb.set_trace()
# # 创建一个进程池，指定进程数
# pool = Pool(96)

# # 使用多进程处理所有文件，并使用tqdm显示进度
# for _ in tqdm(pool.imap(process_file, file_names_cd), total=len(file_names_cd)):
#     pass

# # 关闭进程池，等待所有进程结束
# pool.close()
# pool.join()

index = 0
# # 遍历所有文件名
for file_name in file_names_cd:
    index += 1
    if index < 0:
        continue
    # index += 1
    # 使用imgcat显示图片
    save_path = os.path.join(save_dir, file_name)

    if not os.path.exists(save_path):
        continue
    print(index)
    print(file_name)
    os.system('imgcat {}'.format(save_path))

    # 等待用户按回车键
    input("Press Enter to continue...")
