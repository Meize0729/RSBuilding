# Copyright (c) Open-CD. All rights reserved.
import copy
import os.path as osp
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
import logging

from tqdm import tqdm
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from opencd.registry import DATASETS
from mmengine.fileio import join_path, list_from_file, load
from mmengine.logging import print_log
from mmengine.registry import TRANSFORMS
from mmengine.utils import is_abs

from .basecddataset import BaseCDDataset


@DATASETS.register_module()
class FoundationDataset(BaseDataset):
    
    METAINFO = dict(
    classes=('unchanged / no_building', 'changed / building'),
    palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 data_list: str='', 
                 format_seg_map=None,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.data_list = data_list

        self.format_seg_map = format_seg_map
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        # self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'
                
    def full_init(self):
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        
        # self.type_index = dict()
        # for idx in tqdm(range(len(self.dataset))):
        #     type = self.dataset[idx]['type']
        #     if type in self.type_index.keys():
        #         self.type_index[type].append(idx)
        #     else:
        #         self.type_index[type] = []
        
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True
    
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        # 读取储存具体数据txt的data_list，忽略注释、空行，并去重
        data_list = []
        lines_specific_data_lists = []
        lines = mmengine.list_from_file(
                self.data_list, backend_args=None)
        for line in lines:
            if line.startswith('#') or not line:
                continue
            lines_specific_data_lists.append(line)
        
        # 读取每个data_list，该循环遍历每个数据集的train/val/test的list
        for specific_data_list_path in lines_specific_data_lists: 
            lines = mmengine.list_from_file(
                specific_data_list_path, backend_args=None)
            name = specific_data_list_path.split('/')[-2] + '_' + specific_data_list_path.split('/')[-1].split('.')[0]
            for data_pair in lines:
                img_a, img_b, label_cd, label_a, label_b = data_pair.strip().split('\t')
                data_info = dict(img_a_path=img_a)
                data_info['img_b_path'] = img_b
                data_info['label_cd'] = label_cd
                data_info['label_a'] = label_a
                data_info['label_b'] = label_b

                if data_info['img_b_path'] == '**' and data_info['label_cd'] == '**' and data_info['label_b'] == '**':
                    data_info['type'] = 'only_building_label'
                elif data_info['label_a'] == '**' and data_info['label_b'] == '**':
                    data_info['type'] = 'only_cd_label'
                else:
                    data_info['type'] = 'all_label'
                data_info['dataset_name'] = name

                # no use
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []

                data_list.append(data_info)
        return data_list

    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette


@DATASETS.register_module()
class Bandon_CD_Dataset(BaseCDDataset):
    """Bandon CD dataset"""
    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 format_seg_map='to_binary',
                 ann_file=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            ann_file=ann_file,
            **kwargs)

        self.ann_file = ann_file

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        lines_specific_data_lists = []
        lines = mmengine.list_from_file(
                self.ann_file, backend_args=None)
        for line in lines:
            if line.startswith('#') or not line:
                continue
            lines_specific_data_lists.append(line)
        
        # 读取每个data_list，该循环遍历每个数据集的train/val/test的list
        for specific_data_list_path in lines_specific_data_lists: # /mnt/public/usr/wangmingze/Datasets/CD/BANDON/train.txt
            lines = mmengine.list_from_file(
                specific_data_list_path, backend_args=None)
            name = specific_data_list_path.split('/')[-2] + '_' + specific_data_list_path.split('/')[-1].split('.')[0]
            for data_pair in lines:
                img_a, img_b, label_cd, label_a, label_b = data_pair.strip().split('\t')
                data_info = dict()
                data_info['img_path'] = [img_a, img_b]
                data_info['seg_map_path'] = label_cd
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)

        return data_list

