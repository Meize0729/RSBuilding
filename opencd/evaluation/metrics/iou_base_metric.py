# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from collections import defaultdict

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator.metric import _to_cpu
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from opencd.registry import METRICS


@METRICS.register_module()
class IoU_Base_Metric_Modified(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.results = defaultdict(list)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            a_pred, b_pred, cd_pred = torch.split(data_sample['pred_sem_seg']['data'], [1, 1, 1], dim=0)
            # format_only always for test dataset without ground truth
            if not self.format_only:
                if data_sample['type'] == 'only_cd_label':
                    label = data_sample['gt_sem_seg']['data'].squeeze().to(
                        cd_pred)
                    name2results = self.intersect_and_union(cd_pred.squeeze(), label, num_classes,
                                             self.ignore_index)
                elif data_sample['type'] == 'only_building_label':
                    label = data_sample['gt_sem_seg_from']['data'].squeeze().to(
                        a_pred)
                    
                    name2results = self.intersect_and_union(a_pred.squeeze(), label, num_classes,
                                             self.ignore_index)
                else:
                    label_a = data_sample['gt_sem_seg_from']['data'].squeeze().to(
                        a_pred)
                    label_b = data_sample['gt_sem_seg_to']['data'].squeeze().to(
                        a_pred)
                    label_cd = data_sample['gt_sem_seg']['data'].squeeze().to(
                        a_pred)                    
                    name2results =  [ self.intersect_and_union(a_pred.squeeze(), label_a, num_classes,
                                             self.ignore_index), self.intersect_and_union(b_pred.squeeze(), label_b, num_classes,
                                             self.ignore_index), self.intersect_and_union(cd_pred.squeeze(), label_cd, num_classes,
                                             self.ignore_index) ] 
                self.results[data_sample['dataset_name'] + '_all'].append(name2results) if data_sample['dataset_name'] == 'all_type' else \
                    self.results[data_sample['dataset_name']].append(name2results)
            # format_result
            if self.output_dir is not None:
                name = data_sample['dataset_name']
                basename = osp.splitext(osp.basename(data_sample['img_a_path']))[0]
                dataset_name_dir = osp.abspath(
                    osp.join(self.output_dir, f'{name}'))
                if not osp.exists(dataset_name_dir):
                    os.system(f'mkdir {dataset_name_dir}')
                # # 存image
                # if data_sample['type'] == 'only_cd_label':
                #     img_a_path = data_sample['img_a_path']
                #     img_b_path = data_sample['img_b_path']
                #     T1 = osp.abspath(osp.join(self.output_dir, f'{name}', f'{basename}_T1.png'))
                #     T2 = osp.abspath(osp.join(self.output_dir, f'{name}', f'{basename}_T2.png'))
                #     os.system(f'sudo cp {img_a_path} {T1}')
                #     os.system(f'sudo cp {img_b_path} {T2}')
                # # 存gt
                # if data_sample['type'] == 'only_cd_label' or data_sample['type'] == 'only_building_label':
                #     gt_png_filename = osp.abspath(
                #         osp.join(self.output_dir, f'{name}', f'{basename}_gt.png'))
                #     output_mask = label.cpu().numpy()
                #     output_mask[output_mask != 0] = 255
                #     output = Image.fromarray(output_mask.squeeze().astype(np.uint8))
                #     output.save(gt_png_filename)          
                # save pred        
                for _label, str_label in zip( [a_pred, b_pred, cd_pred], ['a_pred', 'b_pred', 'cd_pred'] ):
                    label = _label
                    if str_label == 'cd_pred':
                        png_filename = osp.abspath(
                            osp.join(self.output_dir, f'{name}', f'{basename}.png'))  
                    else:                     
                        png_filename = osp.abspath(
                            osp.join(self.output_dir, f'{name}', f'{basename}_{str_label}.png'))
                    output_mask = label.cpu().numpy()
                    output_mask[output_mask != 0] = 255
                    output = Image.fromarray(output_mask.squeeze().astype(np.uint8))
                    output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        metrics_class_2 = {key+'_1': ret_metrics_class[key][1] for key in ret_metrics_class if isinstance(ret_metrics_class[key], np.ndarray)}

        return metrics_class_2
    
    def evaluate(self, size: int) -> dict:

        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        print('本次评估数据集共有以下几个:{}'.format(self.results.keys()))
        metrics = [{}]
        for dataset_name, _results in self.results.items():
            results = collect_results(self.results[dataset_name], size, self.collect_device)
            # print(f'\033[31m{dataset_name}\033[0m 本次评估{len(self.results[dataset_name])}个数据')

            if is_main_process():
                # cast all tensors in results list to cpu
                results = _to_cpu(results)
                _metrics = self.compute_metrics(results)  # type: ignore
                # Add prefix to metric names
                _metrics = {
                    '{}_{}'.format(k, dataset_name): v
                    for k, v in _metrics.items()
                }
                if self.prefix:
                    _metrics = {
                        '/'.join((self.prefix, k)): v
                        for k, v in _metrics.items()
                    }
        
                metrics[0].update(_metrics)
            else:
                pass        

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
