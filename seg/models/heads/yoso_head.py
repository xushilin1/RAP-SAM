import math
from typing import Optional, Dict, List, Tuple, Union
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from mmengine.model import ModuleList, caffe2_xavier_init
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.utils import InstanceList, reduce_mean
from mmdet.structures import SampleList
from mmcv.ops import point_sample
from mmdet.models.dense_heads import Mask2FormerHead
import math
from mmengine.model.weight_init import trunc_normal_
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer

from mmengine.dist import get_dist_info

@MODELS.register_module()
class YOSOHead(Mask2FormerHead):
    def __init__(self,
                 num_cls_fcs=1,
                 num_mask_fcs=1,
                 sphere_cls=False,
                 ov_classifier_name=None,
                 use_kernel_updator=False,
                 num_stages=3,
                 feat_channels=256, 
                 out_channels=256, 
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_classes=133,
                 num_queries=100,
                 temperature=0.1, 
                 loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=2.0,
                    reduction='mean',
                    class_weight=[1.0] * 133 + [0.1]),
                 loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                 loss_dice=dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=5.0),  
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_stages = num_stages
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.temperature = temperature

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

        self.kernels = nn.Embedding(self.num_queries, self.feat_channels)

        self.mask_heads = nn.ModuleList()
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead(
                self.num_classes, self.feat_channels, self.num_queries,
                use_kernel_updator=use_kernel_updator,
                sphere_cls=sphere_cls, ov_classifier_name=ov_classifier_name,
                num_cls_fcs=num_cls_fcs, num_mask_fcs=num_mask_fcs
            ))
    
    def init_weights(self) -> None:
        super(AnchorFreeHead, self).init_weights()

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        all_cls_scores = []
        all_masks_preds = []
        proposal_kernels = self.kernels.weight
        object_kernels = proposal_kernels[None].repeat(x.shape[0], 1, 1)
        mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, x)
        
        for stage in range(self.num_stages):
            mask_head = self.mask_heads[stage]
            cls_scores, mask_preds, iou_pred, object_kernels = mask_head(x, object_kernels, mask_preds)
            cls_scores = cls_scores / self.temperature

            all_cls_scores.append(cls_scores)
            all_masks_preds.append(mask_preds)
        
        return all_cls_scores, all_masks_preds

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> Tuple[Tensor]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results
            
class FFN(nn.Module):

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 add_identity=True):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                nn.ReLU(True), 
                nn.Dropout(0.0)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(0.0))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        self.dropout_layer = nn.Dropout(0.0)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class DySepConvAtten(nn.Module):
    def __init__(self, hidden_dim, num_proposals, conv_kernel_size_1d):
        super(DySepConvAtten, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.kernel_size = conv_kernel_size_1d

        self.weight_linear = nn.Linear(self.hidden_dim, self.num_proposals + self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query, value):
        assert query.shape == value.shape
        B, N, C = query.shape
        
        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B, self.num_proposals, 1, self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B, self.num_proposals, self.num_proposals, 1)

        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding='same'))
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')
            res.append(out)

        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        return point_out

class KernelUpdator(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=3,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature):
        """
        Args:
            update_feature (torch.Tensor): [bs, num_proposals, in_channels]
            input_feature (torch.Tensor): [bs, num_proposals, in_channels]
        """
        bs, num_proposals, _ = update_feature.shape
        
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[..., :self.num_params_in]
        param_out = parameters[..., -self.num_params_out:]

        input_feats = self.input_layer(input_feature)
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features

class CrossAttenHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_proposals,
                 frozen_head=False,
                 frozen_pred=False,
                 with_iou_pred=False,
                 sphere_cls=False,
                 ov_classifier_name=None,
                 num_cls_fcs=1,
                 num_mask_fcs=1,
                 conv_kernel_size_1d=3,
                 conv_kernel_size_2d=1,
                 use_kernel_updator=False):
        super(CrossAttenHead, self).__init__()
        self.sphere_cls = sphere_cls
        self.with_iou_pred = with_iou_pred
        self.frozen_head = frozen_head
        self.frozen_pred = frozen_pred
        self.num_cls_fcs = num_cls_fcs
        self.num_mask_fcs = num_mask_fcs
        self.num_classes = num_classes
        self.conv_kernel_size_2d = conv_kernel_size_2d

        self.hidden_dim = in_channels
        self.feat_channels = in_channels
        self.num_proposals = num_proposals
        self.hard_mask_thr = 0.5
        self.use_kernel_updator = use_kernel_updator
        # assert use_kernel_updator
        if use_kernel_updator:
            self.kernel_update = KernelUpdator(
                in_channels=256,
                feat_channels=256,
                out_channels=256,
                input_feat_shape=3,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')
            )
        else:
            self.f_atten = DySepConvAtten(self.feat_channels, self.num_proposals, conv_kernel_size_1d)
            self.f_dropout = nn.Dropout(0.0)
            self.f_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)
            self.k_atten = DySepConvAtten(self.feat_channels, self.num_proposals, conv_kernel_size_1d)
            self.k_dropout = nn.Dropout(0.0)
            self.k_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.s_atten = nn.MultiheadAttention(embed_dim=self.hidden_dim *
                                             self.conv_kernel_size_2d**2,
                                             num_heads=8,
                                             dropout=0.0)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.ffn = FFN(self.hidden_dim, feedforward_channels=2048, num_fcs=2)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

        self.cls_fcs = nn.ModuleList()
        for _ in range(self.num_cls_fcs):
            self.cls_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.cls_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.cls_fcs.append(nn.ReLU(True))

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
            
            # background class
            _dim = cls_embed.size(2)
            _prototypes = cls_embed.size(1)
            if rank == 0:
                back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cuda')
            else:
                back_token = torch.empty(1, _dim, dtype=torch.float32, device='cuda')
            if world_size > 1:
                dist.broadcast(back_token, src=0)
            back_token = back_token.to(device='cpu')
            cls_embed = torch.cat([
                cls_embed, back_token.repeat(_prototypes, 1)[None]
            ], dim=0)
            self.register_buffer('fc_cls', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            # cls embd proj
            cls_embed_dim = self.fc_cls.size(0)
            self.cls_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, cls_embed_dim)
            )

            logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)
        else:
            self.fc_cls = nn.Linear(self.hidden_dim, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(self.num_mask_fcs):
            self.mask_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.mask_fcs.append(nn.ReLU(True))
        self.fc_mask = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.with_iou_pred:
            self.iou_embed = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.apply(self._init_weights)
        if not sphere_cls:
            nn.init.constant_(self.fc_cls.bias, self.bias_value)

        if self.frozen_head:
            self._frozen_head()
        if self.frozen_pred:
            self._frozen_pred()

    def _init_weights(self, m):
        # print("init weights")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _frozen_head(self):
        for n, p in self.kernel_update.named_parameters():
            p.requires_grad = False
        for n, p in self.s_atten.named_parameters():
            p.requires_grad = False
        for n, p in self.s_dropout.named_parameters():
            p.requires_grad = False
        for n, p in self.s_atten_norm.named_parameters():
            p.requires_grad = False
        for n, p in self.ffn.named_parameters():
            p.requires_grad = False
        for n, p in self.ffn_norm.named_parameters():
            p.requires_grad = False

    def _frozen_pred(self):
        # frozen cls_fcs, fc_cls, mask_fcs, fc_mask
        for n, p in self.cls_fcs.named_parameters():
            p.requires_grad = False
        for n, p in self.fc_cls.named_parameters():
            p.requires_grad = False
        for n, p in self.mask_fcs.named_parameters():
            p.requires_grad = False
        for n, p in self.fc_mask.named_parameters():
            p.requires_grad = False

    def train(self, mode):
        super().train(mode)
        if self.frozen_head:
            self.kernel_update.eval()
            self.s_atten.eval()
            self.s_dropout.eval()
            self.s_atten_norm.eval()
            self.ffn.eval()
            self.ffn_norm.eval()
        if self.frozen_pred:
            self.cls_fcs.eval()
            self.fc_cls.eval()
            self.mask_fcs.eval()
            self.fc_mask.eval()

    def forward(self, features, proposal_kernels, mask_preds, self_attn_mask=None):
        B, C, H, W = features.shape

        soft_sigmoid_masks = mask_preds.sigmoid()
        nonzero_inds = soft_sigmoid_masks > self.hard_mask_thr
        hard_sigmoid_masks = nonzero_inds.float()

        # [B, N, C]
        f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, features)
        # [B, N, C, K, K] -> [B, N, C * K * K]
        num_proposals = proposal_kernels.shape[1]
        k = proposal_kernels.view(B, num_proposals, -1)

        # ----
        if self.use_kernel_updator:
            k = self.kernel_update(f, k)
        else:
            f_tmp = self.f_atten(k, f)
            f = f + self.f_dropout(f_tmp)
            f = self.f_atten_norm(f)

            f_tmp = self.k_atten(k, f)
            f = f + self.k_dropout(f_tmp)
            k = self.k_atten_norm(f)

        # [N, B, C]
        k = k.permute(1, 0, 2)

        k_tmp = self.s_atten(query=k, key=k, value=k,attn_mask=self_attn_mask)[0]
        k = k + self.s_dropout(k_tmp)
        k = self.s_atten_norm(k.permute(1, 0, 2))

        obj_feat = self.ffn_norm(self.ffn(k))

        cls_feat = obj_feat
        mask_feat = obj_feat
        
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)

        if self.sphere_cls:
            cls_embd = self.cls_proj(cls_feat) #FIXME Too much cls linear (cls_fcs + cls_proj)
            cls_score = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.fc_cls)
            cls_score = cls_score.max(-1).values
            cls_score = self.logit_scale.exp() * cls_score
        else:
            cls_score = self.fc_cls(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        # [B, N, K * K, C] -> [B, N, C]
        mask_kernels = self.fc_mask(mask_feat)

        new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)
        if self.with_iou_pred:
            iou_pred = self.iou_embed(mask_feat)
            iou_pred = iou_pred
        else:
            iou_pred = None
        return cls_score, new_mask_preds, iou_pred, obj_feat