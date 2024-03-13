import os
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 指定图片所在的文件夹路径
img_dir = '/mnt/public/usr/wangmingze/opencd/pictures/levir-CD_test_nooverlap'

# 指定保存合并后的图片的文件夹路径
output_dir = '/mnt/public/usr/wangmingze/opencd/pictures/levir-CD_test_nooverlap_process'
os.system(f'sudo rm -rf {output_dir}')
os.system(f'mkdir {output_dir}')

# 获取所有图片文件名
img_files = os.listdir(img_dir)

# 按前缀分组图片
img_groups = {}
for img_file in img_files:
    prefix = '_'.join(img_file.split('_')[:4])  # 修改这里，将前缀改为前四部分
    if prefix not in img_groups:
        img_groups[prefix] = []
    img_groups[prefix].append(img_file)

def process_images(prefix, img_group):
    # 按照T1, T2, GT, pred_a, pred_b, pred_cd的顺序排序
    img_group.sort(key=lambda x: ('T1' in x, 'T2' in x, 'gt' in x, 'a_pred' in x, 'b_pred' in x, 'cd_pred' in x))
    img_group = img_group[::-1]

    # 创建一个新的图片，大小为(图片宽度*3 + 间距*4, 图片高度*2 + 间距*3)，背景颜色为淡灰色
    img = Image.open(os.path.join(img_dir, img_group[0]))
    gap = 30  # 间距大小
    new_img = Image.new('RGB', (img.width*3 + gap*4, img.height*2 + gap*4), (211, 211, 211))  # 淡灰色的RGB值为(211, 211, 211)

    # 将6张图片按照要求的顺序排列
    for i, img_file in enumerate(img_group):
        img = Image.open(os.path.join(img_dir, img_file))
        new_img.paste(img, ((i%3)*img.width + (i%3 + 1)*gap, (i//3)*img.height + (i//3 + 1)*gap))

    # 在每张图片下方添加对应的文字
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.truetype('/usr/share/fonts/urw-base35/NimbusRoman-Regular.t1', 25) 
    labels = ['T1_image', 'T2_image', 'GT', 'T1_pred', 'T2_pred', 'CD_pred']
    for i, label in enumerate(labels):
        draw.text(((i%3)*img.width + (i%3 + 1)*gap, (i//3 + 1)*img.height + (i//3 + 1)*gap), label, fill='black', font=font)

    # 保存新的图片
    new_img.save(os.path.join(output_dir, prefix + '.png'))

# 使用线程池进行多线程处理
with ThreadPoolExecutor(max_workers=12) as executor:
    list(tqdm(executor.map(process_images, img_groups.keys(), img_groups.values()), total=len(img_groups)))