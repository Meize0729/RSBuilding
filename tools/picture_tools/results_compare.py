import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

strings = [
    '/mnt/public/usr/wangmingze/Datasets/CD/AerialImageDataset/val/images_512_nooverlap',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/unet/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/deeplabv3/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/segformer-mitb0/vis_data/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/buildformer/compare_pixel',
    '/mnt/public/usr/wangmingze/opencd/pictures_for_bx_ablation/inria/swin_b/compare_pixel',
]


# 这是你要保存图片的路径
save_dir = '/mnt/public/usr/wangmingze/opencd/pictures_tmp/tmp_inria'

# 创建保存路径
os.makedirs(save_dir, exist_ok=True)

# 获取第一个路径下的所有文件名
# s2looking 看到1640了
file_names = os.listdir(os.path.join(strings[0]))

def process_file(file_name):
    # 创建一个新的图像
    fig, axs = plt.subplots(1, 6, figsize=(40, 8))

    flag = False
    # 遍历所有字符串
    for i, string in enumerate(strings):
        # 计算当前图片的行和列
        # row = i // 5
        # col = i % 5

        # 读取图片
        img_path = os.path.join(string, file_name)
        img = mpimg.imread(img_path)
        nonzero_pixels = np.count_nonzero(img)
        total_pixels = img.shape[0] * img.shape[1]
        nonzero_ratio = nonzero_pixels / total_pixels
        if nonzero_ratio < 0.15:
            flag = True
            break

        # 在对应的位置显示图片和字符串
        # axs[row, col].imshow(img)
        # axs[row, col].set_title(string)
        # axs[row, col].axis('off')
    
        axs[i].imshow(img)  # 直接使用索引，因为只有一行
        axs[i].set_title(string.split('/')[-3])
        axs[i].axis('off')

    if flag:
        plt.close(fig)
    else:
        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # 保存图片
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# 创建一个进程池，指定进程数为4
pool = Pool(64)

# 使用多进程处理所有文件，并使用tqdm显示进度
for _ in tqdm(pool.imap(process_file, file_names), total=len(file_names)):
    pass

# 关闭进程池，等待所有进程结束
pool.close()
pool.join()

index = 0
# # 遍历所有文件名
for file_name in file_names:
    index += 1
    # 使用imgcat显示图片
    save_path = os.path.join(save_dir, file_name)

    if not os.path.exists(save_path):
        continue
    print(index)
    print(file_name)
    os.system('imgcat {}'.format(save_path))

    # 等待用户按回车键
    input("Press Enter to continue...")
