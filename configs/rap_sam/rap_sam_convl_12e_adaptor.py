from mmengine.config import read_base

from mmdet.models import BatchFixedSizePad, CrossEntropyLoss, DiceLoss, MaskFormerFusionHead
from mmdet.models.task_modules.assigners import HungarianAssigner, ClassificationCost, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler
from mmdet.models.backbones import ResNet

from seg.models.necks.ramsam_neck import YOSONeck
from seg.models.heads.rapsam_head import RapSAMVideoHead
from seg.models.detectors.rapsam import YOSOVideoSam
from seg.models.backbones import OpenCLIPBackbone
from seg.models.data_preprocessor.vid_sam_preprocessor import VideoPromptDataPreprocessor
with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.coco_panoptic_video_yt19_sam import *
    from .._base_.schedules.schedule_12e import *

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(image_size[1], image_size[0]),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255
    )
]

data_preprocessor = dict(
    type=VideoPromptDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments
)

num_things_classes = 102
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type=YOSOVideoSam,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=OpenCLIPBackbone,
        model_name='convnext_large_d_320',
        fix=True,
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='laion2b_s29b_b131k_ft_soup'
        )
    ),
    neck=dict(
        type=YOSONeck,
        agg_dim=128,
        hidden_dim=256,
        backbone_shape=[256, 512, 1024, 2048],
    ),
    panoptic_head=dict(
        type=RapSAMVideoHead,
        prompt_with_kernel_updator=False,
        panoptic_with_kernel_updator=True,
        use_adaptor=True,
        use_kernel_updator=True,
        sphere_cls=True,
        ov_classifier_name='convnext_large_d_320_Concat_CocoPanopticOVDataset_YouTubeVISDataset_2019',
        num_stages=3,
        feat_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type=MaskFormerFusionHead,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=ClassificationCost, weight=2.0),
                dict(
                    type=CrossEntropyLossCost, weight=5.0, use_sigmoid=True),
                dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type=MaskPseudoSampler)),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
)

val_dataloader = None
val_evaluator = None
val_cfg = None