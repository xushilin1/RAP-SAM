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
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from seg.models.necks import SAMPromptEncoder
from seg.models.utils import preprocess_video_panoptic_gt, mask_pool


@MODELS.register_module()
class Mask2FormerVideoHead(AnchorFreeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_mask_tokens: int = 1,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = None,
                 loss_cls: ConfigType = None,
                 loss_mask: ConfigType = None,
                 loss_dice: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 # ov configs
                 sphere_cls: bool = False,
                 ov_classifier_name: Optional[str] = None,
                 logit: Optional[int] = None,
                 use_adaptor = False,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.use_adaptor = use_adaptor

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.Embedding(num_mask_tokens, feat_channels)
        self.pb_embedding = nn.Embedding(2, feat_channels)
        self.pos_linear = nn.Linear(2 * feat_channels, feat_channels)

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        # assert pixel_decoder.encoder.layer_cfg. \
        #            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding3D(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        if not sphere_cls:
            self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.iou_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, 1))

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

        # prepare OV things
        # OV cls embed
        if sphere_cls:
            rank, world_size = get_dist_info()
            if ov_classifier_name is None:
                _dim = 1024  # temporally hard code
                cls_embed = torch.empty(self.num_classes, _dim)
                torch.nn.init.orthogonal_(cls_embed)
                cls_embed = cls_embed[:, None]
            else:
                ov_path = os.path.join(os.path.expanduser('~/.cache/embd'), f"{ov_classifier_name}.pth")
                cls_embed = torch.load(ov_path)
                cls_embed_norm = cls_embed.norm(p=2, dim=-1)
                assert torch.allclose(cls_embed_norm, torch.ones_like(cls_embed_norm))
            if self.loss_cls and self.loss_cls.use_sigmoid:
                pass
            else:
                _dim = cls_embed.size(2)
                _prototypes = cls_embed.size(1)

                if rank == 0:
                    back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cuda')
                    # back_token = back_token / back_token.norm(p=2, dim=-1, keepdim=True)
                else:
                    back_token = torch.empty(1, _dim, dtype=torch.float32, device='cuda')
                if world_size > 1:
                    dist.broadcast(back_token, src=0)
                back_token = back_token.to(device='cpu')
                cls_embed = torch.cat([
                    cls_embed, back_token.repeat(_prototypes, 1)[None]
                ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            # cls embd proj
            cls_embed_dim = self.cls_embed.size(0)
            self.cls_proj = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, cls_embed_dim)
            )

            # Haobo Yuan:
            # For the logit_scale, I refer to this issue.
            # https://github.com/openai/CLIP/issues/46#issuecomment-945062212
            # https://github.com/openai/CLIP/issues/46#issuecomment-782558799
            # Based on my understanding, it is a mistake of CLIP.
            # Because they mention that they refer to InstDisc (Wu, 2018) paper.
            # InstDisc set a non-learnable temperature to np.log(1 / 0.07).
            # 4.6052 is np.log(1 / 0.01)
            # np.log(1 / 0.07) will be fast converged to np.log(1 / 0.01)
            if logit is None:
                logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            else:
                logit_scale = torch.tensor(logit, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)

            # Mask Pooling
            self.mask_pooling = mask_pool
            self.mask_pooling_proj = nn.Sequential(
                nn.LayerNorm(feat_channels),
                nn.Linear(feat_channels, feat_channels)
            )

        if use_adaptor:
            cross_attn_cfg = dict(embed_dims=256, batch_first=True, num_heads=8)
            self.panoptic_attn = MultiheadAttention(**cross_attn_cfg)
            self.panoptic_norm = nn.LayerNorm(256)
            if sphere_cls:
                cls_embed_dim = self.cls_embed.size(0)
                self.panoptic_cls = nn.Sequential(
                    nn.Linear(feat_channels, cls_embed_dim)
                )
            else:
                raise NotImplementedError
            self.prompt_attn = MultiheadAttention(**cross_attn_cfg)
            self.prompt_norm = nn.LayerNorm(256)
            self.prompt_iou = nn.Linear(256, 1)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward_logit(self, cls_embd):
        cls_pred = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      num_frames: int = 0) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
                - num_frames: How many frames are there in video.
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        if isinstance(self.cls_embed, nn.Module):
            cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        if not isinstance(self.cls_embed, nn.Module):
            maskpool_embd = self.mask_pooling(x=mask_feature, mask=mask_pred.detach())
            maskpool_embd = self.mask_pooling_proj(maskpool_embd)
            cls_embd = self.cls_proj(maskpool_embd + decoder_out)
            cls_pred = self.forward_logit(cls_embd)

        iou_pred = self.iou_embed(decoder_out)

        if num_frames > 0:
            assert len(mask_pred.shape) == 4
            assert mask_pred.shape[2] % num_frames == 0
            frame_h = mask_pred.shape[2] // num_frames
            num_q = mask_pred.shape[1]
            _mask_pred = mask_pred.unflatten(-2, (num_frames, frame_h)).flatten(1, 2)
            attn_mask = F.interpolate(
                _mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
            attn_mask = attn_mask.unflatten(1, (num_q, num_frames)).flatten(2, 3)
        else:
            attn_mask = F.interpolate(
                mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, iou_pred, attn_mask

    def forward(self, x: List[Tensor], batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
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
        batch_size = len(batch_img_metas)
        #(bs_nf, c, h,w)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        if num_frames > 0:
            mask_features = mask_features.unflatten(0, (batch_size, num_frames))
            mask_features = mask_features.transpose(1, 2).flatten(2, 3) #(bs, c, nf*h,w)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i]) #(bs_nf, c, h,w)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1) #(bs_nf,h*w, c)
            if num_frames > 0:
                decoder_input = decoder_input.unflatten(0, (batch_size, num_frames))
                decoder_input = decoder_input.flatten(1, 2) #(bs, nf*h*w, c)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1 if num_frames == 0 else num_frames
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input) #(bs, nf*h*w, c)
            decoder_positional_encodings.append(decoder_positional_encoding) #(bs, nf*h*w, c)

        if self.prompt_training:
            query_feat, input_query_bbox, self_attn_mask, _ = self.prepare_for_dn_mo(
                batch_data_samples)
            query_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
            query_embed = self.pos_linear(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
            query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))
            self_attn_mask = None

        cls_pred_list = []
        mask_pred_list = []
        iou_pred_list = []
        cls_pred, mask_pred, iou_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames
        )
        cls_pred_list.append(cls_pred)
        iou_pred_list.append(iou_pred)
        if num_frames > 0: #(bs, 100, nf*h, w)-->(bs, 100, nf, h, w)
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat, #(bs, 100, c)
                key=decoder_inputs[level_idx], #(bs, nf*h*w, c)
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, iou_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames
            )

            cls_pred_list.append(cls_pred)
            iou_pred_list.append(iou_pred)
            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)
        
        if self.use_adaptor:
            keys = mask_features.flatten(2).transpose(1, 2).contiguous()
            h, w = mask_features.shape[-2] // num_frames_real, mask_features.shape[-1]
            mask = decoder_input.new_zeros((batch_size, num_frames_real, h, w), dtype=torch.bool)
            key_pos = self.decoder_positional_encoding(mask)
            key_pos = key_pos.transpose(1, 2).flatten(2).permute(0, 2, 1)
            if not self.prompt_training:
                object_kernels = self.panoptic_attn(query_feat, keys, key_pos=key_pos, query_pos=query_embed)
                object_kernels = self.panoptic_norm(object_kernels)
                mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                
                cls_embd = self.panoptic_cls(object_kernels)
                cls_scores = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
                cls_scores = cls_scores.max(-1).values
                cls_scores = self.logit_scale.exp() * cls_scores
                
                if num_frames > 0: 
                    mask_pred_list.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    mask_pred_list.append(mask_preds)
                cls_pred_list.append(cls_scores)
                iou_pred_list.append(iou_pred_list[-1])
            else:
                object_kernels = self.prompt_attn(query_feat, keys, key_pos=key_pos, query_pos=query_embed)
                object_kernels = self.prompt_norm(object_kernels)
                iou_preds = self.prompt_iou(object_kernels)
                mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, mask_features)
                
                if num_frames > 0: 
                    mask_pred_list.append(mask_preds.unflatten(2, (num_frames, -1)))
                else:
                    mask_pred_list.append(mask_preds)
                cls_pred_list.append(cls_pred_list[-1])
                iou_pred_list.append(iou_preds)

        return cls_pred_list, mask_pred_list, iou_pred_list, query_feat

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                return_query=False,
                ) -> Tuple[Tensor, ...]:
        """Test without augmentaton.

        Args:
            return_query:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        self.prompt_training = False
        data_sample = batch_data_samples[0]
        if isinstance(data_sample, TrackDataSample):
            img_shape = data_sample[0].metainfo['batch_input_shape']
            num_frames = len(data_sample)
        else:
            if 'gt_instances_collected' in data_sample:
                self.prompt_training = True
            img_shape = data_sample.metainfo['batch_input_shape']
            num_frames = 0
        all_cls_scores, all_mask_preds, all_iou_preds, query_feat = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        iou_results = all_iou_preds[-1]

        if num_frames > 0:
            mask_pred_results = mask_pred_results.flatten(1, 2)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)
        if num_frames > 0:
            num_queries = mask_cls_results.shape[1]
            mask_pred_results = mask_pred_results.unflatten(1, (num_queries, num_frames))

        if return_query:
            return mask_cls_results, mask_pred_results, query_feat, iou_results
        else:
            return mask_cls_results, mask_pred_results, iou_results

    def prepare_for_dn_mo(self, batch_data_samples):
        scalar, noise_scale = 100, 0.4
        gt_instances = [t.gt_instances_collected for t in batch_data_samples]

        point_coords = torch.stack([inst.point_coords for inst in gt_instances])
        pb_labels = torch.stack([inst['pb_labels'] for inst in gt_instances])
        labels = torch.zeros_like(pb_labels).long()

        boxes = point_coords  # + boxes

        factors = []
        for i, data_sample in enumerate(batch_data_samples):
            h, w, = data_sample.metainfo['img_shape']
            factor = boxes[i].new_tensor([w, h, w, h]).unsqueeze(0).repeat(boxes[i].size(0), 1)
            factors.append(factor)
        factors = torch.stack(factors, 0)

        boxes = bbox_xyxy_to_cxcywh(boxes / factors)  # xyxy / factor or xywh / factor ????
        # box_start = [t['box_start'] for t in targets]
        box_start = [len(point) for point in point_coords]

        known_labels = labels
        known_pb_labels = pb_labels
        known_bboxs = boxes

        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # add very small noise to input points; no box
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            known_bbox_expand += torch.mul(
                (torch.rand_like(known_bbox_expand) * 2 - 1.0),
                diff) * noise_scale

            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        input_label_embed = self.pb_embedding(known_pb_labels_expaned)

        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(
            self.num_mask_tokens,
            1) + self.mask_tokens.weight.unsqueeze(0).repeat(
                input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(
            self.num_mask_tokens, 1)

        single_pad = self.num_mask_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_mask_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'pad_size': pad_size,
            'scalar': scalar,
        }
        return input_query_label, input_query_bbox, attn_mask, mask_dict