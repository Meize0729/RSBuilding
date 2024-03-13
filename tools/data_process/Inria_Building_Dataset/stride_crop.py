import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu

import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


Building = np.array([255, 255, 255])
Clutter = np.array([0, 0, 0])
num_classes = 2


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-img-dir",   default="data_dir/AerialImageDataset/train_images")
    parser.add_argument("--input-mask-dir",  default="data_dir/AerialImageDataset/train_masks")
    parser.add_argument("--output-img-dir",  default="data_dir/AerialImageDataset/train_processed/images")
    parser.add_argument("--output-mask-dir", default="data_dir/AerialImageDataset/train_processed/gt")

    parser.add_argument("--mode", type=str, default='val')
    parser.add_argument("--split-size-h", type=int, default=512)
    parser.add_argument("--split-size-w", type=int, default=512)
    parser.add_argument("--stride-h", type=int, default=512)
    parser.add_argument("--stride-w", type=int, default=512)
    return parser.parse_args()


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = Building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = Clutter

    return mask_rgb


def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 255
    label_seg[np.all(label == Clutter, axis=-1)] = 0

    return label_seg


def image_augment(image, mask, mode='train'):
    image_list = []
    mask_list = []
    image_width, image_height = image.shape[1], image.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        # train_transform = [
        #     # albu.HorizontalFlip(p=0.5),
        #     # albu.VerticalFlip(p=0.5),
        #     # albu.RandomRotate90(p=0.5),
        #     # albu.RandomSizedCrop(min_max_height=(image_height//2, image_height),
        #     #                      width=image_width, height=image_height, p=0.15),
        #     # albu.RandomShadow(num_shadows_lower=2, num_shadows_upper=3,
        #     #                   shadow_dimension=3, shadow_roi=(0, 0.5, 1, 1), p=0.1),
        #     # albu.GaussianBlur(p=0.01),
        #     albu.OneOf([
        #         albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        #         albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25)
        #     ], p=0.15)
        # ]
        # aug = albu.Compose(train_transform)(image=image.copy(), mask=mask.copy())

        image_list_train = [image]
        mask_list_train = [mask]
        for i in range(len(image_list_train)):
            mask_tmp = rgb2label(mask_list_train[i])
            image_list.append(image_list_train[i])
            mask_list.append(mask_tmp)
    else:
        mask = rgb2label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def padifneeded(image, mask, patch_size, stride):

    oh, ow = image.shape[0], image.shape[1]
    padh, padw = 0, 0
    while (oh + padh -patch_size[0]) % stride[0] != 0:
        padh = padh + 1
    while (ow + padw -patch_size[1]) % stride[1] != 0:
        padw = padw + 1

    h, w = oh + padh, ow + padw

    pad = albu.PadIfNeeded(min_height=h, min_width=w)(image=image, mask=mask)
    img_pad, mask_pad = pad['image'], pad['mask']

    return img_pad, mask_pad


def patch_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, mode, split_size, stride) = inp
    # if mode == 'val':
    #     gt_path = masks_output_dir + "_gt"
    #     if not os.path.exists(gt_path):
    #         os.makedirs(gt_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    id = os.path.splitext(os.path.basename(img_path))[0]
    assert img.shape == mask.shape

    img, mask = padifneeded(img.copy(), mask.copy(), split_size, stride)

    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), mode=mode)
    assert len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
        for y in range(0, img.shape[0], stride[0]):
            for x in range(0, img.shape[1], stride[1]):
                img_tile_cut = img[y:y + split_size[0], x:x + split_size[1]]
                mask_tile_cut = mask[y:y + split_size[0], x:x + split_size[1]]
                img_tile, mask_tile = img_tile_cut, mask_tile_cut

                if img_tile.shape[0] == split_size[0] and img_tile.shape[1] == split_size[1] \
                        and mask_tile.shape[0] == split_size[0] and mask_tile.shape[1] == split_size[1]:
                    # bins = np.array(range(num_classes + 1))
                    # class_pixel_counts, _ = np.histogram(mask_tile, bins=bins)
                    # cf = class_pixel_counts / (mask_tile.shape[0] * mask_tile.shape[1])
                    pixels_255 = np.sum(mask_tile == 255)
                    total_pixels = mask_tile.size
                    ratio = pixels_255 / total_pixels
                    if ratio > 0.05 and mode == 'train':
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)
                    elif ratio > 0 and mode == 'val':
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(id, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    input_img_dir = args.input_img_dir
    input_mask_dir = args.input_mask_dir
    img_paths = glob.glob(os.path.join(input_img_dir, "*.tif"))
    mask_paths = glob.glob(os.path.join(input_mask_dir, "*.tif"))
    img_paths.sort()
    mask_paths.sort()

    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    mode = args.mode

    # os.system(f'sudo rm -rf {imgs_output_dir}')
    # os.system(f'sudo rm -rf {masks_output_dir}')

    split_size_h = args.split_size_h
    split_size_w = args.split_size_w
    split_size = (split_size_h, split_size_w)
    stride_h = args.stride_h
    stride_w = args.stride_w
    stride = (stride_h, stride_w)

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, mode, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    # for tmp in inp:
    #     patch_format(tmp)
    mpp.Pool(processes=16).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))