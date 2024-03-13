import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import tifffile

# img = tifffile.imread('/mnt/public/usr/wangmingze/Datasets/CD/DSIFN/test/mask/0.tif')
# img = cv2.imread('/mnt/public/usr/wangmingze/Datasets/CD/MSBC/train/label_512/681.png')
# # 寻找最大像素值及其坐标
# max_value = img.max()
# img[img == 1] = 255
# # 打印最大像素值及其坐标
# print('Max value:', max_value)
# cv2.imwrite('/mnt/public/usr/wangmingze/Datasets/CD/MSBC/train/label_512/681_new.png', img)
# os.system('imgcat /mnt/public/usr/wangmingze/Datasets/CD/MSBC/train/label_512/681_new.png')

# import numpy as np
# img = cv2.imread('/mnt/public/usr/wangmingze/Datasets/CD/test/targets/hurricane-florence_00000005_post_disaster_target.png', cv2.IMREAD_UNCHANGED)
# img[img != 0] = 255
# cv2.imwrite('/mnt/public/usr/wangmingze/Datasets/CD/test/targets/hurricane-florence_00000005_post_disaster_target_2.png', img)

# un = np.unique(img)
# from PIL import Image, ImageDraw
# from shapely.wkt import loads

# # 创建一个1024x1024的全0图像
# img = Image.new('L', (1024, 1024), 0)
# draw = ImageDraw.Draw(img)

# # 解析多边形的坐标
# wkt_str = "POLYGON ((406.8541728120606 0, 419.9457992893496 1.301514173001342, 437.9223610193584 1.496911583110133, 441.8303092215343 7.358833886373868, 447.4814117528476 9.101988215511403, 410.5156481854635 106.187894727652, 405.6410420007536 108.625197820007, 370.7063643436653 94.40759644793623, 376.3934048924936 76.12782325527377, 380.8617938951444 77.75269198351043, 386.1426172619136 62.31643906526213, 380.0493595310261 59.87913597290713, 385.3301828977953 46.06775178289549, 392.6420921748602 48.50505487525049, 398.7353499057477 33.47501913906135, 394.6731780851561 32.25636759288385, 406.8541728120606 0))"
# polygon = loads(wkt_str)
# coords = list(polygon.exterior.coords)
# # 在图像上绘制多边形
# draw.polygon(coords, fill=255)
# # 显示图像
# img.save('/mnt/public/usr/wangmingze/Datasets/CD/test/targets/hurricane-florence_00000005_pre_disaster_target_3.png')


import json
from PIL import Image, ImageDraw
from shapely.wkt import loads
# with open('/mnt/public/usr/wangmingze/Datasets/CD/test/labels/hurricane-florence_00000005_post_disaster.json', 'r') as f:
#     infos = json.load(f)

# img_shape = (infos['metadata']['height'], infos['metadata']['width'])
# img = Image.new('L', img_shape, 0)
# draw = ImageDraw.Draw(img)
# info_list = infos['features']['xy']
# wkt_list = []
# for info in info_list:
#     if info['properties']['feature_type'] == 'building' and info['properties']['subtype'] != 'no-damage':
#         wkt_list.append(info['wkt'])
# for wkt_str in wkt_list:
#     polygon = loads(wkt_str)
#     coords = list(polygon.exterior.coords)
#     draw.polygon(coords, fill=255)
# img.save('/mnt/public/usr/wangmingze/Datasets/CD/test/targets/hurricane-florence_00000005_post_disaster_target_3.png')

def json2png(filename):
    real_path = os.path.join(folder_path, filename)
    name = filename.split('.')[0]
    flag = True if 'post' in filename else False
    with open(real_path, 'r') as f:
        infos = json.load(f)
    if infos['metadata']['height'] < 512 or infos['metadata']['width'] < 512:
        return None
    img_shape = (infos['metadata']['height'], infos['metadata']['width'])
    info_list = infos['features']['xy']
    if flag:
        img_b = Image.new('L', img_shape, 0)
        draw_b = ImageDraw.Draw(img_b)
        img_cd = Image.new('L', img_shape, 0)
        draw_cd = ImageDraw.Draw(img_cd)        
        wkt_list_b = []
        wkt_list_cd = []
        for info in info_list:
            if info['properties']['feature_type'] == 'building':
                if info['properties']['subtype'] != 'no-damage':
                    wkt_list_cd.append(info['wkt'])
                elif info['properties']['subtype'] == 'no-damage':
                    wkt_list_b.append(info['wkt'])
        for wkt_str_cd in wkt_list_cd:
            polygon = loads(wkt_str_cd)
            coords = list(polygon.exterior.coords)
            draw_cd.polygon(coords, fill=255)
        for wkt_str_b in wkt_list_b:
            polygon = loads(wkt_str_b)
            coords = list(polygon.exterior.coords)
            draw_b.polygon(coords, fill=255)
        cd_name = name.replace('_post_disaster', '')
        img_cd.save(f'{out_path_cd}/{cd_name}.png')
        img_b.save(f'{out_path_ab}/{name}.png')
    else:
        img_a = Image.new('L', img_shape, 0)
        draw_a = ImageDraw.Draw(img_a)
        wkt_list_a = []
        for info in info_list:
            if info['properties']['feature_type'] == 'building':
                wkt_list_a.append(info['wkt'])
        for wkt_str_a in wkt_list_a:
            polygon = loads(wkt_str_a)
            coords = list(polygon.exterior.coords)
            draw_a.polygon(coords, fill=255)    
        img_a.save(f'{out_path_ab}/{name}.png')


    
if __name__ == '__main__':
    # 文件夹路径
    folder_path = '/mnt/public/usr/wangmingze/Datasets/CD/xView2_hold/labels'
    real_folder_path = folder_path.split('/')[-1]
    out_path_ab = folder_path.replace(real_folder_path, 'AB')
    out_path_cd = folder_path.replace(real_folder_path, 'CD_label')
    os.system(f'rm -rf {out_path_ab}')
    os.system(f'mkdir {out_path_ab}')
    os.system(f'rm -rf {out_path_cd}')
    os.system(f'mkdir {out_path_cd}')

    # 获取文件夹下所有的文件路径
    filenames = [filename for filename in os.listdir(folder_path)]
    
    # 使用多进程并行处理图像
    with Pool(32) as p:
        list(tqdm(p.imap(json2png, filenames), total=len(filenames)))


import pdb; pdb.set_trace()
