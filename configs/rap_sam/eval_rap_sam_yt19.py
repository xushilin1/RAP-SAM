from mmengine import read_base
from seg.models.detectors import Mask2formerVideoMinVIS
from mmdet.models import BatchFixedSizePad
from seg.models.data_preprocessor.vidseg_data_preprocessor import VideoSegDataPreprocessor

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.youtube_vis_2019 import *
    from .._base_.schedules.schedule_12e import *
    from .rap_sam_r50_12e_adaptor import model

model_name = 'yoso_r18_adaptor'
val_evaluator = dict(
    type=YouTubeVISMetric,
    metric='youtube_vis_ap',
    outfile_prefix=f'./youtube_vis_2019_results_{model_name}',
    format_only=True
)

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
    type=VideoSegDataPreprocessor,
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

num_things_classes = 40
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

ov_datasets_name = 'YouTubeVISDataset_2019'
default_hooks.update(
    logger=dict(type=LoggerHook, interval=1),
)



test_evaluator = val_evaluator

ov_model_name = 'convnext_large_d_320'
model.update(
    data_preprocessor=data_preprocessor,
    type=Mask2formerVideoMinVIS,
    clip_size=5,
    clip_size_small=3,
    whole_clip_thr=0,
    small_clip_thr=15,
    overlap=0,
    panoptic_head=dict(
        ov_classifier_name=f'{ov_model_name}_{ov_datasets_name}',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
    ),
)