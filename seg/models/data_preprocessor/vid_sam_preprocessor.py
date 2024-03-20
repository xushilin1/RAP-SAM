# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor
from torch import Tensor
import torch.nn.functional as F
from mmdet.structures.bbox import BaseBoxes
from mmengine.model.utils import stack_batch

from mmdet.models.utils.misc import samplelist_boxtype2tensor, unfold_wo_center
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample, TrackSampleList
from mmdet.structures.mask import BitmapMasks
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import PixelData

try:
    import skimage
except ImportError:
    skimage = None

from .vidseg_data_preprocessor import VideoSegDataPreprocessor
from mmdet.models.data_preprocessors import TrackDataPreprocessor
@MODELS.register_module()
class VideoPromptDataPreprocessor(VideoSegDataPreprocessor):
    """Image pre-processor for tracking tasks.

        Accepts the data sampled by the dataloader, and preprocesses
        it into the format of the model input. ``TrackDataPreprocessor``
        provides the tracking data pre-processing as follows:

        - Collate and move data to the target device.
        - Pad inputs to the maximum size of current batch with defined
          ``pad_value``. The padding size can be divisible by a defined
          ``pad_size_divisor``
        - Stack inputs to inputs.
        - Convert inputs from bgr to rgb if the shape of input is (1, 3, H, W).
        - Normalize image with defined std and mean.
        - Do batch augmentations during training.
        - Record the information of ``batch_input_shape`` and ``pad_shape``.

        Args:
            mean (Sequence[Number], optional): The pixel mean of R, G, B
                channels. Defaults to None.
            std (Sequence[Number], optional): The pixel standard deviation of
                R, G, B channels. Defaults to None.
            pad_size_divisor (int): The size of padded image should be
                divisible by ``pad_size_divisor``. Defaults to 1.
            pad_value (Number): The padded pixel value. Defaults to 0.
            pad_mask (bool): Whether to pad instance masks. Defaults to False.
            mask_pad_value (int): The padded pixel value for instance masks.
                Defaults to 0.
            bgr_to_rgb (bool): whether to convert image from BGR to RGB.
                Defaults to False.
            rgb_to_bgr (bool): whether to convert image from RGB to RGB.
                Defaults to False.
            use_det_processor: (bool): whether to use DetDataPreprocessor
                in training phrase. This is mainly for some tracking models
                fed into one image rather than a group of image in training.
                Defaults to False.
    .       boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
                bboxes data to ``Tensor`` type. Defaults to True.
            batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 use_det_processor: bool = False,
                 **kwargs):
        super().__init__(mean=mean, std=std, **kwargs)
        self.use_det_processor = use_det_processor
        if mean is not None and not self.use_det_processor:
            self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1), False)
            self.register_buffer('std',  torch.tensor(std).view(1, -1, 1, 1), False)

    def forward(self, data: dict, training: bool = False) -> Dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``TrackDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], OptSampleList]: Data in the
            same format as the model input.
        """
        inputs, data_samples = data['inputs'], data['data_samples']
        from mmengine.utils import is_seq_of
        if data_samples[0].get('data_tag', 'coco') == 'sam':
            batch_pad_shape = self._get_pad_shape(data)
            data = self.cast_data(data)  # type: ignore
            _batch_inputs = data['inputs']
            if is_seq_of(_batch_inputs, torch.Tensor):
                batch_inputs = []
                for _batch_input in _batch_inputs:
                    if self._channel_conversion:
                        _batch_input = _batch_input[[2, 1, 0], ...]
                    _batch_input = _batch_input.float()
                    if self._enable_normalize:
                        _batch_input = (_batch_input - self.mean[0]) / self.std[0]
                    batch_inputs.append(_batch_input)
                batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor, self.pad_value)
            data['inputs'] = batch_inputs
            data.setdefault('data_samples', None)

            # super(VideoSegDataPreprocessor, self).forward
            inputs, data_samples = data['inputs'], data['data_samples']

            if data_samples is not None:
                batch_input_shape = tuple(inputs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)

                if self.pad_mask and training:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg and training:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    inputs, data_samples = batch_aug(inputs, data_samples)

            return {'inputs': inputs, 'data_samples': data_samples}
        else:
            return super().forward(data, training)