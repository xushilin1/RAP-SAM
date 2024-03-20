# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmdet.models import Mask2FormerTransformerDecoder
from mmengine.dist import get_dist_info
from mmengine.model import caffe2_xavier_init, ModuleList
from mmengine.structures import InstanceData, PixelData
from torch import Tensor
from mmdet.models.layers import MLP, inverse_sigmoid
from mmdet.models.layers import coordinate_to_encoding
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList, TrackDataSample
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)
from mmdet.models.layers import SinePositionalEncoding3D
from mmdet.models.utils import multi_apply, preprocess_panoptic_gt, get_uncertain_point_coords_with_randomness
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from seg.models.necks import SAMPromptEncoder
from seg.models.utils import preprocess_video_panoptic_gt, mask_pool

from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from .mask2former_vid import Mask2FormerVideoHead
from .yoso_head import CrossAttenHead, KernelUpdator

@MODELS.register_module()
class RapSAMVideoHead(Mask2FormerVideoHead):

    def __init__(self,
                 frozen_head=False,
                 frozen_pred=False,
                 use_adaptor=False,
                 prompt_with_kernel_updator=False,
                 panoptic_with_kernel_updator=False,
                 num_mask_tokens = 1,
                 num_stages = 3,
                 use_kernel_updator=False,
                 sphere_cls = False,
                 ov_classifier_name = None,
                 temperature=0.1,
                 feat_channels=256,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 loss_cls: ConfigType = None,
                 loss_mask: ConfigType = None,
                 loss_dice: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 matching_whole_map: bool = False,
                 enable_box_query: bool = False,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.prompt_with_kernel_updator = prompt_with_kernel_updator
        self.panoptic_with_kernel_updator = panoptic_with_kernel_updator
        self.use_adaptor = use_adaptor

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Embedding(num_mask_tokens, feat_channels)
        self.pb_embedding = nn.Embedding(2, feat_channels)
        self.pos_linear = nn.Linear(2 * feat_channels, feat_channels)

        self.matching_whole_map = matching_whole_map
        self.enable_box_query = enable_box_query

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.num_stages = num_stages
        self.kernels = nn.Embedding(self.num_queries, feat_channels)
        self.mask_heads = nn.ModuleList()
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead(
                self.num_classes, self.feat_channels, self.num_queries,
                use_kernel_updator=use_kernel_updator,
                frozen_head=frozen_head, frozen_pred=frozen_pred,
                sphere_cls=sphere_cls,
                ov_classifier_name=ov_classifier_name, with_iou_pred=True))
        self.temperature = temperature

        if use_adaptor:
            cross_attn_cfg = dict(embed_dims=256, batch_first=True, num_heads=8)
            if self.panoptic_with_kernel_updator:
                self.panoptic_attn = KernelUpdator(feat_channels=256)
                self.panoptic_norm = nn.Identity()
                if sphere_cls:
                    cls_embed_dim = self.mask_heads[0].fc_cls.size(0)
                    self.panoptic_cls = nn.Sequential(
                        nn.Linear(feat_channels, cls_embed_dim)
                    )
                else:
                    raise NotImplementedError
                    self.panoptic_cls = nn.Linear(256, self.num_classes+1)
            else:
                self.panoptic_attn = MultiheadAttention(**cross_attn_cfg)
                self.panoptic_norm = nn.LayerNorm(256)
                if sphere_cls:
                    cls_embed_dim = self.mask_heads[0].fc_cls.size(0)
                    self.panoptic_cls = nn.Sequential(
                        nn.Linear(feat_channels, cls_embed_dim)
                    )
                else:
                    raise NotImplementedError
                    self.panoptic_cls = nn.Linear(256, self.num_classes+1)
            
            if self.prompt_with_kernel_updator:
                self.prompt_attn = KernelUpdator(feat_channels=256)
                self.prompt_norm = nn.Identity()
                self.prompt_iou = nn.Linear(256, 1)
            else:
                self.prompt_attn = MultiheadAttention(**cross_attn_cfg)
                self.prompt_norm = nn.LayerNorm(256)
                self.prompt_iou = nn.Linear(256, 1)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        

    def init_weights(self) -> None:
        pass
    
    def forward(self, x, batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        batch_img_metas = []
        if isinstance(batch_data_samples[0], TrackDataSample):
            for track_sample in batch_data_samples:
                cur_list = []
                for det_sample in track_sample:
                    cur_list.append(det_sample.metainfo)
                batch_img_metas.append(cur_list)
            num_frames = len(batch_img_metas[0])
        else:
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
            num_frames = 0
        bs = len(batch_img_metas)
        
        all_cls_scores = []
        all_masks_preds = []
        all_iou_preds = []
        if self.prompt_training:
            input_query_label, input_query_bbox, self_attn_mask, mask_dict = self.prepare_for_dn_mo(
                batch_data_samples)
            pos_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
            pos_embed = self.pos_linear(pos_embed)
            object_kernels = input_query_label + pos_embed
        else:
            object_kernels = self.kernels.weight[None].repeat(bs, 1, 1)
            self_attn_mask = None
        mask_features = x
        if num_frames > 0: # (bs*num_frames, c, h, w) -> (bs, c, num_frames*h, w)
            mask_features = mask_features.unflatten(0, (bs, num_frames))
            mask_features = mask_features.transpose(1, 2).flatten(2, 3)
        
        mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
        for stage in range(self.num_stages):
            mask_head = self.mask_heads[stage]
            cls_scores, mask_preds, iou_preds, object_kernels = mask_head(
                mask_features, object_kernels, mask_preds, self_attn_mask)
            cls_scores = cls_scores / self.temperature
            all_iou_preds.append(iou_preds)
            all_cls_scores.append(cls_scores)
            if num_frames > 0: 
                #(bs,num_query, num_frames*h, w) --> (bs,num_query,num_frames,h,w)
                all_masks_preds.append(mask_preds.unflatten(2, (num_frames, -1)))
            else:
                all_masks_preds.append(mask_preds)
        
        if self.use_adaptor:
            keys = mask_features.flatten(2).transpose(1, 2).contiguous()
            if not self.prompt_training:
                if self.panoptic_with_kernel_updator:
                    hard_sigmoid_masks = (mask_preds.sigmoid() > 0.5).float()
                    f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, mask_features)
                    object_kernels = self.panoptic_attn(f, object_kernels)
                    object_kernels = self.panoptic_norm(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                else:
                    object_kernels = self.panoptic_attn(object_kernels, keys)
                    object_kernels = self.panoptic_norm(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                cls_embd = self.panoptic_cls(object_kernels)
                cls_scores = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.mask_heads[0].fc_cls)
                cls_scores = cls_scores.max(-1).values
                cls_scores = self.mask_heads[0].logit_scale.exp() * cls_scores
                
                if num_frames > 0: 
                    all_masks_preds.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    all_masks_preds.append(mask_preds)
                all_cls_scores.append(cls_scores)
                all_iou_preds.append(all_iou_preds[-1])
            else:
                if self.prompt_with_kernel_updator:
                    hard_sigmoid_masks = (mask_preds.sigmoid() > 0.5).float()
                    f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, mask_features)
                    object_kernels = self.prompt_attn(f, object_kernels)
                    object_kernels = self.prompt_norm(object_kernels)
                    iou_preds = self.prompt_iou(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                else:
                    object_kernels = self.prompt_attn(object_kernels, keys)
                    object_kernels = self.prompt_norm(object_kernels)
                    iou_preds = self.prompt_iou(object_kernels)
                    mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                if num_frames > 0: 
                    all_masks_preds.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    all_masks_preds.append(mask_preds)
                all_cls_scores.append(all_cls_scores[-1])
                all_iou_preds.append(iou_preds)
        return all_cls_scores, all_masks_preds, all_iou_preds, object_kernels

    def _loss_by_feat_single(self, cls_scores, mask_preds, iou_preds, batch_gt_instances, batch_img_metas):
        batch_size, num_ins = cls_scores.size(0), cls_scores.size(1)
        if self.prompt_training:
            num_imgs = mask_preds.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
            mask_targets = torch.cat([item.masks for item in batch_gt_instances])
            mask_weights = mask_targets.new_ones((batch_size, num_ins), dtype=torch.float)
            avg_factor = cls_scores.size(1)

            num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
            num_total_masks = max(num_total_masks, 1)

            mask_preds = mask_preds[mask_weights > 0]

            if mask_targets.shape[0] == 0:
                # zero match
                loss_dice = mask_preds.sum()
                loss_mask = mask_preds.sum()
                loss_iou = loss_dice.sum() * 0.0
                loss_cls = cls_scores.sum() * 0.0
                return loss_cls, loss_mask, loss_dice, loss_iou

            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_preds.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                mask_point_targets = point_sample(
                    mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)

            mask_point_preds = point_sample(mask_preds.unsqueeze(1),
                                            points_coords).squeeze(1)

            # dice loss
            loss_mask = self.loss_mask(mask_point_preds,
                                    mask_point_targets,
                                    reduction_override='none').mean(1)
            loss_dice = self.loss_dice(mask_point_preds,
                                    mask_point_targets,
                                    reduction_override='none')

            iou_preds = iou_preds.flatten()  # (bs, 60, 6) --> (bs, 360)
            iou_target = 1 - (loss_dice / self.loss_dice.loss_weight)
            loss_iou = F.mse_loss(iou_preds, iou_target, reduction="none")
            loss_mask = loss_mask.sum() / num_total_masks
            loss_dice = loss_dice.sum() / num_total_masks
            loss_iou = loss_iou.sum() / num_total_masks * 10.0

            loss_cls = cls_scores.sum() * 0.0 + self.kernels.weight.sum() * 0.0
            if self.use_adaptor:
                for n, p in self.panoptic_attn.named_parameters():
                    loss_cls += p.sum() * 0.0
                for n, p in self.panoptic_norm.named_parameters():
                    loss_cls += p.sum() * 0.0
                for n, p in self.panoptic_cls.named_parameters():
                    loss_cls += p.sum() * 0.0
            return loss_cls, loss_mask, loss_dice, loss_iou
        else:
            cls_scores_list = [cls_scores[i] for i in range(batch_size)]
            mask_preds_list = [mask_preds[i] for i in range(batch_size)]
            labels_list, label_weights_list, mask_targets_list, mask_weights_list, avg_factor = \
                self.get_targets(cls_scores_list, mask_preds_list, batch_gt_instances, batch_img_metas)
            labels = torch.stack(labels_list, dim=0)
            label_weights = torch.stack(label_weights_list, dim=0)
            mask_targets = torch.cat(mask_targets_list, dim=0)
            mask_weights = torch.stack(mask_weights_list, dim=0)

        
            # classification loss
            # shape (batch_size * num_queries, )
            cls_scores = cls_scores.flatten(0, 1)
            labels = labels.flatten(0, 1)
            label_weights = label_weights.flatten(0, 1)
            class_weight = cls_scores.new_tensor(self.class_weight)
            ignore_inds = labels.eq(-1.)
            # zero will not be involved in the loss cal
            labels[ignore_inds] = 0
            label_weights[ignore_inds] = 0.
        
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                label_weights,
                # avg_factor=cls_avg_factor
                avg_factor=class_weight[labels].sum()
            )
        
            # loss_mask
            num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
            num_total_masks = max(num_total_masks, 1)
            # extract positive ones
            # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
            mask_preds = mask_preds[mask_weights > 0]

            if mask_targets.shape[0] == 0:
                # zero match
                loss_dice = mask_preds.sum()
                loss_mask = mask_preds.sum()
                loss_iou = iou_preds.sum() * 0.0
                loss_iou += (self.mask_tokens.weight.sum() + self.pb_embedding.weight.sum()) * 0.0
                loss_iou += (self.pos_linear.weight.sum() + self.pos_linear.bias.sum()) * 0.0
                if self.use_adaptor:
                    for n, p in self.prompt_attn.named_parameters():
                        loss_iou += p.sum() * 0.0
                    for n, p in self.prompt_norm.named_parameters():
                        loss_iou += p.sum() * 0.0
                    for n, p in self.prompt_iou.named_parameters():
                        loss_iou += p.sum() * 0.0
                return loss_cls, loss_mask, loss_dice, loss_iou

            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_preds.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                mask_point_targets = point_sample(
                    mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
            # shape (num_queries, h, w) -> (num_queries, num_points)
            mask_point_preds = point_sample(
                mask_preds.unsqueeze(1), points_coords).squeeze(1)
            # dice loss
            loss_dice = self.loss_dice(
                mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

            # mask loss
            # shape (num_queries, num_points) -> (num_queries * num_points, )
            mask_point_preds = mask_point_preds.reshape(-1)
            # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
            mask_point_targets = mask_point_targets.reshape(-1)
            loss_mask = self.loss_mask(
                mask_point_preds,
                mask_point_targets,
                avg_factor=num_total_masks * self.num_points
            )
            loss_iou = iou_preds.sum() * 0.0
            loss_iou += (self.mask_tokens.weight.sum() + self.pb_embedding.weight.sum()) * 0.0
            loss_iou += (self.pos_linear.weight.sum() + self.pos_linear.bias.sum()) * 0.0
            if self.use_adaptor:
                for n, p in self.prompt_attn.named_parameters():
                    loss_iou += p.sum() * 0.0
                for n, p in self.prompt_norm.named_parameters():
                    loss_iou += p.sum() * 0.0
                for n, p in self.prompt_iou.named_parameters():
                    loss_iou += p.sum() * 0.0
            return loss_cls, loss_mask, loss_dice, loss_iou
