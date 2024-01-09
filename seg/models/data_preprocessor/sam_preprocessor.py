import torch
import random
from mmdet.models import DetDataPreprocessor
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from kornia.contrib import distance_transform
import torch.nn.functional as F

@MODELS.register_module()
class SAMDataPreprocessor(DetDataPreprocessor):
    def __init__(self, *args, num_mask_tokens=6, repeat=False, num_proposals=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_mask_tokens = num_mask_tokens
        self.repeat = repeat
        self.num_proposals = num_proposals

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']
        if training:
            return dict(inputs=inputs, data_samples=data_samples)
        for data_sample in data_samples:
            gt_instances = data_sample.gt_instances
            
            device = gt_instances.labels.device
            gt_collected = []
            
            ori_num_instances = len(gt_instances)
            ori_indices = torch.randperm(ori_num_instances)
            if self.repeat and ori_num_instances < self.num_proposals:
                repeat_cnt = (self.num_proposals // ori_num_instances) + 1
                ori_indices = ori_indices.repeat(repeat_cnt)
                ori_indices = ori_indices[:self.num_proposals]
            gt_instances.masks = gt_instances.masks.to_tensor(torch.bool, device)
            h, w = data_sample.metainfo['img_shape']
            for instance_idx in ori_indices:
                mask = gt_instances.masks[instance_idx]
                mask_clone = mask[:h, :w][None, None, :]
                n, _, h, w = mask_clone.shape
                mask_dt = (distance_transform((~F.pad(mask_clone, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :, 1:-1, 1:-1])
                selected_point = torch.tensor([mask_dt.argmax()/w, mask_dt.argmax()%w]).long().flip(0).to(device)
                selected_point = torch.cat([selected_point - 3, selected_point + 3], 0)
                # selected_point = gt_instances.bboxes[instance_idx]
                gt_collected.append({
                    'point_coords': selected_point,
                    'instances': None,
                    'masks': mask,
                })
            data_sample.gt_instances_collected = InstanceData(
                point_coords=torch.stack([itm['point_coords'] for itm in gt_collected]),
                sub_instances=[itm['instances'] for itm in gt_collected],
                masks=torch.stack([itm['masks'] for itm in gt_collected]),
            )
            pb_labels = torch.ones(len(data_sample.gt_instances_collected), dtype=torch.long, device=device)
            data_sample.gt_instances_collected.pb_labels = pb_labels
        return dict(inputs=inputs, data_samples=data_samples)