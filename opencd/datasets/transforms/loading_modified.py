# Copyright (c) Open-CD. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiImgLoadImageFromFile_Modified(MMCV_LoadImageFromFile):

    def __init__(self, **kwargs) -> None:
         super().__init__(**kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        data_type = results['type']
        if data_type == 'only_building_label':
            try:
                filenames = [results['img_a_path']]
            except:
                filenames = [results['img_path']]
        else:
            filenames = [results['img_a_path'], results['img_b_path']]
        imgs = []
        try:
            for filename in filenames:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    img = img.astype(np.float32)
                imgs.append(img)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # results['img'] = imgs
        # results['img_shape'] = imgs[0].shape[:2]
        # results['ori_shape'] = imgs[0].shape[:2]
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        return results
    
    
@TRANSFORMS.register_module()
class MultiImgMultiAnnLoadAnnotations_Modified(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_semantic_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_semantic_zero_label = reduce_semantic_zero_label
        if self.reduce_semantic_zero_label is not None:
            warnings.warn('`reduce_semantic_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_semantic_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """


        if results['type'] == 'only_cd_label':
            img_bytes = fileio.get(
                results['label_cd'], backend_args=self.backend_args)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='grayscale', # in mmseg: unchanged
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
            gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1

            results['gt_seg_map'] = gt_semantic_seg
            results['seg_fields'].extend(['gt_seg_map',])

        elif results['type'] == 'only_building_label':
            img_bytes_from = fileio.get(
                results['label_a'], backend_args=self.backend_args)
            # img_bytes_from = fileio.get(
            #     results['seg_map_path'], backend_args=self.backend_args)
            gt_semantic_seg_from = mmcv.imfrombytes(
                img_bytes_from, flag='grayscale',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            # gt_semantic_seg_from = mmcv.imfrombytes(
            #     img_bytes_from, flag='unchanged',
            #     backend=self.imdecode_backend).squeeze().astype(np.uint8)
            gt_semantic_seg_from_copy = gt_semantic_seg_from.copy()
            gt_semantic_seg_from[gt_semantic_seg_from_copy < 128] = 0
            gt_semantic_seg_from[gt_semantic_seg_from_copy >= 128] = 1

            # results['gt_seg_map_from'] = gt_semantic_seg_from
            # results['seg_fields'].extend(['gt_seg_map_from'])
            results['gt_seg_map_from'] = gt_semantic_seg_from
            results['seg_fields'].extend(['gt_seg_map_from'])

        else:
            img_bytes = fileio.get(
                results['label_cd'], backend_args=self.backend_args)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='grayscale', # in mmseg: unchanged
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            img_bytes_from = fileio.get(
                results['label_a'], backend_args=self.backend_args)
            gt_semantic_seg_from = mmcv.imfrombytes(
                img_bytes_from, flag='grayscale',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            img_bytes_to = fileio.get(
                results['label_b'], backend_args=self.backend_args)
            gt_semantic_seg_to = mmcv.imfrombytes(
                img_bytes_to, flag='grayscale',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
            gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            gt_semantic_seg_to_copy = gt_semantic_seg_to.copy()
            gt_semantic_seg_to[gt_semantic_seg_to_copy < 128] = 0
            gt_semantic_seg_to[gt_semantic_seg_to_copy >= 128] = 1
            gt_semantic_seg_from_copy = gt_semantic_seg_from.copy()
            gt_semantic_seg_from[gt_semantic_seg_from_copy < 128] = 0
            gt_semantic_seg_from[gt_semantic_seg_from_copy >= 128] = 1  

            results['gt_seg_map'] = gt_semantic_seg
            results['gt_seg_map_from'] = gt_semantic_seg_from
            results['gt_seg_map_to'] = gt_semantic_seg_to
            results['seg_fields'].extend(['gt_seg_map', 
            'gt_seg_map_from', 'gt_seg_map_to'])
        


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_semantic_zero_label={self.reduce_semantic_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str