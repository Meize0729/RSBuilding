# Copyright (c) Open-CD. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample
from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiImgPackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'seg_map_path_from', 
                            'seg_map_path_to', 'ori_shape','img_shape', 'dataset_name'
                            'pad_shape', 'scale_factor', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img
            
            imgs = [_transform_img(img) for img in results['img']]
            imgs = torch.cat(imgs, axis=0) # -> (6, H, W)
            packed_results['inputs'] = imgs

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))
        
        if 'gt_seg_map_from' in results:
            gt_sem_seg_data_from = dict(
                data=to_tensor(results['gt_seg_map_from'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_from=PixelData(**gt_sem_seg_data_from)))

        if 'gt_seg_map_to' in results:
            gt_sem_seg_data_to = dict(
                data=to_tensor(results['gt_seg_map_to'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_to=PixelData(**gt_sem_seg_data_to)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class MultiImgPackSegInputs_Modified(BaseTransform):

    def __init__(self,
                 meta_keys=('img_a_path', 'img_b_path', 'label_cd', 'label_a', 
                            'label_b', 'type', 'ori_shape','img_shape', 'dataset_name' ,
                            'pad_shape', 'scale_factor', 'flip',
                            'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            # results['img'] = [results['img']]
            def _transform_img(img):
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if not img.flags.c_contiguous:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                else:
                    img = img.transpose(2, 0, 1)
                    img = to_tensor(img).contiguous()
                return img
            imgs = [_transform_img(img) for img in results['img']]
            if len(imgs) == 2:
                imgs = torch.cat(imgs, axis=0) # -> (6, H, W)
            else:
                imgs = imgs[0]
            packed_results['inputs'] = imgs

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))
        
        if 'gt_seg_map_from' in results:
            gt_sem_seg_data_from = dict(
                data=to_tensor(results['gt_seg_map_from'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_from=PixelData(**gt_sem_seg_data_from)))

        if 'gt_seg_map_to' in results:
            gt_sem_seg_data_to = dict(
                data=to_tensor(results['gt_seg_map_to'][None,
                                                     ...].astype(np.int64)))
            data_sample.set_data(dict(gt_sem_seg_to=PixelData(**gt_sem_seg_data_to)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            else:
                img_meta[key] = None
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
