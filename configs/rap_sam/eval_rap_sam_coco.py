from mmengine import read_base
from mmengine.config import read_base

from mmdet.models import BatchFixedSizePad

from seg.models.data_preprocessor.vidseg_data_preprocessor import VideoSegDataPreprocessor

with read_base():
    from .._base_.datasets.coco_panoptic_lsj import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_12e import *
    from .rap_sam_r50_12e_adaptor import model
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

test_dataloader.update(
    batch_size=1
)
ov_model_name = 'convnext_large_d_320'
ov_datasets_name = 'CocoPanopticOVDataset'
num_things_classes = 80
num_stuff_classes = 53
model.update(
    data_preprocessor=data_preprocessor,
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
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
    ),
)
