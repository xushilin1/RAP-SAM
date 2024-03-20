from typing import Any, List, Optional, Sequence, Union

import torch
from torch import Tensor

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)

from mmengine.fileio import dump
from mmengine.logging import print_log
from mmengine.registry import METRICS

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from mmengine.evaluator import BaseMetric

from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet.registry import METRICS
from mmdet.evaluation.metrics import CocoPanopticMetric
from mmengine.structures import BaseDataElement
from mmengine.evaluator.metric import _to_cpu

@METRICS.register_module()
class InteractiveEvaluator(BaseMetric):

    def __init__(self,
                 iou_iter=1,
                 max_clicks=20,
                 num_tokens=6,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_iter = iou_iter
        self.num_tokens = num_tokens
        self.max_clicks = max_clicks
        self.iou_list = []
        # self.oracle_iou_list = []
        self.num_samples = 0
        self.all_ious = [0.5, 0.8, 0.85, 0.9]

    def get_iou(self, gt_masks, pred_masks, ignore_label=-1):
        n, h, w = pred_masks.shape
        import torch.nn.functional as F
        gt_masks = F.interpolate(gt_masks.float().unsqueeze(0), size=(h, w), mode='bilinear').bool().squeeze(0)
        rev_ignore_mask = ~(gt_masks == ignore_label)
        intersection = ((gt_masks & pred_masks) & rev_ignore_mask).reshape(n, h * w).sum(dim=-1)
        union = ((gt_masks | pred_masks) & rev_ignore_mask).reshape(n, h * w).sum(dim=-1)
        ious = (intersection / union)
        return ious

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_masks = data_sample['pred_instances']['masks']
            # all_pred_mask = data_sample['all_pred_instances']['masks']

            gt_masks = data_sample['gt_instances_collected']['masks']
            self.iou_list.extend(self.get_iou(gt_masks, pred_masks))
            # gt_masks_repeat = gt_masks.repeat_interleave(self.num_tokens, 0)

            # iou_all = self.get_iou(gt_masks_repeat, all_pred_mask)
            # selected_ious, index = iou_all.view(-1, self.num_tokens).max(1)
            # self.oracle_iou_list.extend(selected_ious)

    def compute_noc(self, iou_list):
        def _get_noc(iou_arr, iou_thr):
            vals = iou_arr >= iou_thr
            return 1 if vals else self.max_clicks

        noc_list = {}
        for iou_thr in self.all_ious:
            scores_arr = [
                _get_noc(iou_arr, iou_thr) for iou_arr in iou_list
            ]
            
            noc_list[str(iou_thr)] = scores_arr

        iou_before_max_iter = iou_list

        num_samples = len(iou_list)
        pred_noc =  {}
        for key, value in noc_list.items():
            pred_noc[key] = sum(value) * 1.0 / len(value)
        pred_noc['iou_max_iter'] = sum(iou_before_max_iter) / num_samples

        return pred_noc

    def compute_metrics(self, iou_list, oracle_iou_list) -> Dict[str, float]:

        pred_noc = self.compute_noc(iou_list)
        # oracle_pred_noc = self.compute_noc(oracle_iou_list)

        results = {}
        for idx in range(len(self.all_ious)):
            result_str = 'noc@{}'.format(self.all_ious[idx])
            results[result_str] = pred_noc[str(self.all_ious[idx])]
        results['miou@iter{}'.format(self.iou_iter)] = pred_noc['iou_max_iter']

        # for idx in range(len(self.all_ious)):
        #     result_str = 'oracle_noc@{}'.format(self.all_ious[idx])
            # results[result_str] = oracle_pred_noc[str(self.all_ious[idx])]
        # results['oracle_miou@iter{}'.format(self.iou_iter)] = oracle_pred_noc['iou_max_iter']
        return results

    def evaluate(self, size: int) -> dict:
        iou_list = collect_results(self.iou_list, size, self.collect_device)
        # oracle_iou_list = collect_results(self.oracle_iou_list, size, self.collect_device)
        oracle_iou_list = None
        if is_main_process():
            iou_list = _to_cpu(iou_list)
            # oracle_iou_list = _to_cpu(oracle_iou_list)
            _metrics = self.compute_metrics(iou_list, oracle_iou_list)  # type: ignore
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        self.iou_list = []
        # self.oracle_iou_list = []

        # reset the results list
        return metrics[0]