# dataset settings
from mmengine import read_base
from mmengine.dataset import DefaultSampler, RepeatDataset

from seg.datasets.concat_dataset import ConcatOVDataset
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler

with read_base():
    from .coco_panoptic_video_lsj import train_dataloader as _coco_vid_train_loader
    from .youtube_vis_2019 import train_dataloader as _yt19_train_loader
    from .coco_panoptic_lsj import val_dataloader, val_evaluator, test_dataloader, test_evaluator
    from .youtube_vis_2019 import image_size
from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.datasets.pipeliens.formatting import GeneratePoint
from seg.datasets.pipeliens.loading import LoadPanopticAnnotationsAll
data_root = 'data/coco/'
backend_args = None
image_size = (1280, 736)

num_mask_tokens = 1

sam_train_pipeline = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(type=LoadPanopticAnnotationsAll),
    dict(type=RandomFlip, prob=0.5),
    dict(type=RandomResize, resize_type=Resize, scale=image_size, ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type=RandomCrop, crop_size=image_size, crop_type='absolute', recompute_bbox=True, allow_negative_crop=True),
    dict(type=FilterAnnotationsHB, by_box=False, by_mask=True, min_gt_mask_area=32),
    dict(type=PackDetInputs),
    dict(type=GeneratePoint, num_proposals=30, num_mask_tokens=num_mask_tokens)
]

sam_dataset=dict(
    type=CocoPanopticOVDataset,
    data_root=data_root,
    ann_file='annotations/panoptic_train2017.json',
    data_prefix=dict(img='train2017/', seg='annotations/panoptic_train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sam_train_pipeline,
    backend_args=backend_args
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=VideoSegAspectRatioBatchSampler),
    dataset=dict(
        type=ConcatOVDataset,
        data_tag=('coco', 'yt19', 'sam'),
        datasets=[
            dict(
                type=RepeatDataset,
                dataset=_coco_vid_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_yt19_train_loader.dataset,
                times=25,
            ),
            dict(
                type=RepeatDataset,
                dataset=sam_dataset,
                times=1,
            )
        ]
    ),
)
