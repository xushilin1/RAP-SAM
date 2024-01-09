from mmengine import read_base
from mmengine.config import read_base

from mmdet.models import BatchFixedSizePad
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.dataset import DefaultSampler
from seg.models.data_preprocessor.vidseg_data_preprocessor import VideoSegDataPreprocessor
from seg.evaluation.metrics.interactive_evaluation import InteractiveEvaluator
from seg.models.data_preprocessor import SAMDataPreprocessor
from seg.datasets.pipeliens.loading import LoadPanopticAnnotationsAll
from seg.datasets.coco_ov import CocoPanopticOVDataset
from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.datasets.pipeliens.formatting import GeneratePoint
from seg.datasets.pipeliens.loading import LoadPanopticAnnotationsAll
from mmcv.transforms import LoadImageFromFile, RandomResize

with read_base():
    from .._base_.datasets.coco_panoptic_lsj import train_dataloader
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_12e import *
    from .rap_sam_r50_12e_adaptor import model


batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(1024, 1024),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255
    )
]
data_preprocessor = dict(
    type=SAMDataPreprocessor,
    num_mask_tokens=1,
    repeat=False,
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
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadPanopticAnnotationsAll),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(
        type=PackDetInputs,
    )
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CocoPanopticOVDataset,
        serialize_data=False,
        data_root='data/coco/',
        ann_file='annotations/panoptic_val2017.json',
        data_prefix=dict(img='val2017/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader
val_evaluator = [
    dict(
        type=InteractiveEvaluator,
        num_tokens=1,
    )
]
test_evaluator = val_evaluator

num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes

ov_datasets_name = 'CocoPanopticOVDataset'
ov_model_name = 'convnext_large_d_320'

model.update(
    data_preprocessor=data_preprocessor,
    inference_sam=True,
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