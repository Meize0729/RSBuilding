# Copyright (c) Open-CD. All rights reserved.
import copy
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image.geometric import _scale_size
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_list_of, is_seq_of, is_str, is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from opencd.registry import TRANSFORMS

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None
    
@TRANSFORMS.register_module()
class MultiImgRandomCrop_Modified(BaseTransform):

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_type: list=[],
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.ignore_type = ignore_type

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img'][0]
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1. and results['type'] not in self.ignore_type:
            # Repeat 50 times
            for _ in range(50):
                try:
                    seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                except:
                    seg_temp = self.crop(results['gt_seg_map_from'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # if picture_from.shape == self.crop_size; do nothing
        if results['img'][0].shape[:2] == self.crop_size:
            return results
        crop_bbox = self.crop_bbox(results)

        # crop the image
        imgs = [self.crop(img, crop_bbox) for img in results['img']]

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = imgs
        results['img_shape'] = imgs[0].shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
    
    
@TRANSFORMS.register_module()
class MultiImgRandomRotate_Modified(BaseTransform):

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    @cache_randomness
    def generate_degree(self):
        return np.random.rand() < self.prob, np.random.uniform(
            min(*self.degree), max(*self.degree))

    def transform(self, results: dict) -> dict:
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        rotate, degree = self.generate_degree()
        if results['ori_shape'] == (512, 512):
            return results
        if rotate:
            # rotate image
            results['img'] = [
                mmcv.imrotate(
                    img,
                    angle=degree,
                    border_value=self.pal_val,
                    center=self.center,
                    auto_bound=self.auto_bound) for img in results['img']]

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str
    

@TRANSFORMS.register_module()
class RandomMosaic_Modified(BaseTransform):

    def __init__(self,
                 prob,
                 img_scale=(512,512),
                 center_ratio_range=(0.5, 1.5),
                 ignore_type=['only_cd_label', 'all_label'],
                 pad_val=0,
                 seg_pad_val=255):
        assert 0 <= prob and prob <= 1
        assert isinstance(img_scale, tuple)
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        
        self.ignore_type = ignore_type

    @cache_randomness
    def do_mosaic(self):
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        mosaic = self.do_mosaic() and results['type'] not in self.ignore_type
        if mosaic:
            results = self._mosaic_transform_img(results)
            results = self._mosaic_transform_seg(results)
        return results

    def get_indices(self, dataset: MultiImageMixDataset, type_list) -> list:
        pass

    @cache_randomness
    def generate_mosaic_center(self):
        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        return center_x, center_y

    def _mosaic_transform_img(self, results: dict) -> dict:

        assert 'mix_results' in results
        mosaic_imgs = []
        self.center_x, self.center_y = self.generate_mosaic_center()
        center_position = (self.center_x, self.center_y)
        if len(results['img'][0].shape) == 3:
            c = results['img'][0].shape[2]
            mosaic_img = np.full(
                (int(self.img_scale[0]), int(self.img_scale[1]), c),
                self.pad_val,
                dtype=results['img'][0].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0]), int(self.img_scale[1])),
                self.pad_val,
                dtype=results['img'][0].dtype)
        if len(results['img']) == 2:
            mosaic_img_2 = mosaic_img
            
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = copy.deepcopy(results)
            else:
                result_patch = copy.deepcopy(results['mix_results'][i - 1])
                
            x1, y1, x2, y2 = self._mosaic_combine(loc, center_position)

            img_i = result_patch['img'][0]
            for i in range(len(result_patch['img'])):
                if i == 0:
                    mosaic_img[y1:y2, x1:x2] = result_patch['img'][i][y1:y2, x1:x2]
                else:
                    mosaic_img_2[y1:y2, x1:x2] = result_patch['img'][i][y1:y2, x1:x2]
                

        mosaic_imgs.append(mosaic_img) 
        if len(results['img']) == 2: 
            mosaic_imgs.append(mosaic_img_2)

        results['img'] = mosaic_imgs
        results['img_shape'] = mosaic_imgs[0].shape
        results['ori_shape'] = mosaic_imgs[0].shape

        return results

    def _mosaic_transform_seg(self, results: dict) -> dict:

        assert 'mix_results' in results
        for key in results.get('seg_fields', []):
            mosaic_seg = np.full(
                (int(self.img_scale[0]), int(self.img_scale[1])),
                self.seg_pad_val,
                dtype=results[key].dtype)

            # mosaic center x, y
            center_position = (self.center_x, self.center_y)

            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    result_patch = copy.deepcopy(results)
                else:
                    result_patch = copy.deepcopy(results['mix_results'][i - 1])

                gt_seg_i = result_patch[key]
                x1, y1, x2, y2 = self._mosaic_combine(loc, center_position)
                mosaic_seg[y1:y2, x1:x2] = gt_seg_i[y1:y2, x1:x2] 


            results[key] = mosaic_seg

        return results

    def _mosaic_combine(self, loc: str, center_position_xy: Sequence[float]) -> tuple:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        if loc == 'top_left':
            x1, y1, x2, y2 = 0, 0, center_position_xy[0], center_position_xy[1]
        elif loc == 'top_right':
            x1, y1, x2, y2 = center_position_xy[0], 0, 512, center_position_xy[1]
        elif loc == 'bottom_left':
            x1, y1, x2, y2 = 0, center_position_xy[1], center_position_xy[0], 512
        else:
            x1, y1, x2, y2 = center_position_xy[0], center_position_xy[1], 512, 512
            
        return x1, y1, x2, y2

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'seg_pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class MultiImgRandomResize_Modified(BaseTransform):

    def __init__(
        self,
        ratio_range: Tuple[float, float] = None,
        prob = 0.5,
        resize_type: str = 'MultiImgResize',
        **resize_kwargs,
    ) -> None:

        self.ratio_range = ratio_range
        self.prob = prob

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.
        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.
        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.
        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.
        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    @cache_randomness
    def _random_scale(self) -> tuple:
        """Private function to randomly sample an scale according to the type
        of ``scale``.
        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        if is_tuple_of(self.scale, int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range)
        elif is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')

        return scale

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, ``img``, ``gt_semantic_seg``,
            ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        # import pdb; pdb.set_trace()
        tmp_scale = results['img_shape']
        self.scale = tmp_scale
        if np.random.rand() > self.prob or tmp_scale == (512, 512):
            results['scale'] = tmp_scale
        else:
            results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        # import pdb; pdb.set_trace()
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str
    