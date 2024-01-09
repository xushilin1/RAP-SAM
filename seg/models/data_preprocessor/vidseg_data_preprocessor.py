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


@MODELS.register_module()
class VideoSegDataPreprocessor(DetDataPreprocessor):
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
            # overwrite the ``register_bufffer`` in ``ImgDataPreprocessor``
            # since the shape of ``mean`` and ``std`` in tracking tasks must be
            # (T, C, H, W), which T is the temporal length of the video.
            self.register_buffer('mean',
                                 torch.tensor(mean).view(1, -1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(1, -1, 1, 1), False)

        self.ins_seg_gen = BoxInstMaskGenerator()

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
        if not isinstance(data['data_samples'][0], TrackDataSample):
            use_det = True
        else:
            use_det = False
        if self.use_det_processor and training or use_det:
            batch_pad_shape = self._get_pad_shape(data)
        else:
            batch_pad_shape = self._get_track_pad_shape(data)

        data = self.cast_data(data)
        imgs, data_samples = data['inputs'], data['data_samples']

        if self.use_det_processor and training or use_det:
            assert imgs[0].dim() == 3, \
                'Only support the 3 dims when use detpreprocessor in training'
            if self._channel_conversion:
                imgs = [_img[[2, 1, 0], ...] for _img in imgs]
            # Convert to `float`
            imgs = [_img.float() for _img in imgs]
            if self._enable_normalize:
                imgs = [(_img - self.mean.squeeze(0)) / self.std.squeeze(0) for _img in imgs]
            inputs = stack_batch(imgs, self.pad_size_divisor, self.pad_value)
        else:
            assert imgs[0].dim() == 4, \
                'Only support the 4 dims when use trackprocessor in training'
            # The shape of imgs[0] is (T, C, H, W).
            channel = imgs[0].size(1)
            if self._channel_conversion and channel == 3:
                imgs = [_img[:, [2, 1, 0], ...] for _img in imgs]
            # change to `float`
            imgs = [_img.float() for _img in imgs]
            if self._enable_normalize:
                imgs = [(_img - self.mean) / self.std for _img in imgs]
            inputs = stack_track_batch(imgs, self.pad_size_divisor,
                                       self.pad_value)

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs.size()[-2:])
            if self.use_det_processor and training or use_det:
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
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
            else:
                for track_data_sample, pad_shapes in zip(
                        data_samples, batch_pad_shape):
                    for i in range(len(track_data_sample)):
                        det_data_sample = track_data_sample[i]
                        det_data_sample.set_metainfo({
                            'batch_input_shape': batch_input_shape,
                            'pad_shape': pad_shapes[i]
                        })

                if self.boxtype2tensor:
                    tracking_samplelist_boxtype2tensor(data_samples)

                if self.pad_mask and training:
                    self.pad_track_gt_masks(data_samples)

                if self.pad_seg and training:
                    self.pad_track_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                if self.use_det_processor and training or use_det:
                    inputs, data_samples = batch_aug(inputs, data_samples)
                else:
                    # For video segmentation, the batch augmentation are conducted
                    # on the batch dimension only, which means it will be run several
                    # times given the number of frames.
                    final_inputs = []
                    for frame_id in range(inputs.size(1)):
                        det_data_samples = [
                            track_data_sample[frame_id]
                            for track_data_sample in data_samples
                        ]
                        aug_inputs, aug_det_samples = batch_aug(
                            inputs[:, frame_id], det_data_samples)
                        final_inputs.append(aug_inputs.unsqueeze(1))
                        for track_data_sample, det_sample in zip(
                                data_samples, aug_det_samples):
                            track_data_sample.video_data_samples[frame_id] = det_sample
                    inputs = torch.cat(final_inputs, dim=1)

        # Note: inputs may contain large number of frames, so we must make
        # sure that the mmeory is contiguous for stable forward
        inputs = inputs.contiguous()

        if training and use_det and 'masks' not in data_samples[0].gt_instances:
            assert 'bboxes' in data_samples[0].gt_instances
            return self.ins_seg_gen(dict(inputs=inputs, data_samples=data_samples), mean=self.mean, std=self.std)
        return dict(inputs=inputs, data_samples=data_samples)

    def _get_track_pad_shape(self, data: dict) -> Dict[str, List]:
        """Get the pad_shape of each image based on data and pad_size_divisor.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            Dict[str, List]: The shape of padding.
        """
        batch_pad_shape = dict()
        batch_pad_shape = []
        for imgs in data['inputs']:
            # The sequence images in one sample among a batch have the same
            # original shape
            pad_h = int(np.ceil(imgs.shape[-2] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(imgs.shape[-1] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_shapes = [(pad_h, pad_w)] * imgs.size(0)
            batch_pad_shape.append(pad_shapes)
        return batch_pad_shape

    def pad_track_gt_masks(self,
                           data_samples: Sequence[TrackDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in data_samples[0][0].get('gt_instances', None):
            for track_data_sample in data_samples:
                for i in range(len(track_data_sample)):
                    det_data_sample = track_data_sample[i]
                    masks = det_data_sample.gt_instances.masks
                    # TODO: whether to use BitmapMasks
                    assert isinstance(masks, BitmapMasks)
                    batch_input_shape = det_data_sample.batch_input_shape
                    det_data_sample.gt_instances.masks = masks.pad(
                        batch_input_shape, pad_val=self.mask_pad_value)

    def pad_track_gt_sem_seg(self,
                             data_samples: Sequence[TrackDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in data_samples[0][0]:
            for track_data_sample in data_samples:
                for i in range(len(track_data_sample)):
                    det_data_sample = track_data_sample[i]
                    gt_sem_seg = det_data_sample.gt_sem_seg.sem_seg
                    h, w = gt_sem_seg.shape[-2:]
                    pad_h, pad_w = det_data_sample.batch_input_shape
                    gt_sem_seg = F.pad(
                        gt_sem_seg,
                        pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                        mode='constant',
                        value=self.seg_pad_value)
                    det_data_sample.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)


def stack_track_batch(tensors: List[torch.Tensor],
                      pad_size_divisor: int = 0,
                      pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the common height and width
    is divisible by ``pad_size_divisor``. The difference between this function
    and ``stack_batch`` in MMEngine is that this function can process batch
    sequence images with shape (N, T, C, H, W).

    Args:
        tensors (List[Tensor]): The input multiple tensors. each is a
            TCHW 4D-tensor. T denotes the number of key/reference frames.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the common height and width is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need a divisibility of 32. Defaults to 0
        pad_value (int, float): The padding value. Defaults to 0

    Returns:
       Tensor: The NTCHW 5D-tensor. N denotes the batch size.
    """
    assert isinstance(tensors, list), \
        f'Expected input type to be list, but got {type(tensors)}'
    assert len(set([tensor.ndim for tensor in tensors])) == 1, \
        f'Expected the dimensions of all tensors must be the same, ' \
        f'but got {[tensor.ndim for tensor in tensors]}'
    assert tensors[0].ndim == 4, f'Expected tensor dimension to be 4, ' \
                                 f'but got {tensors[0].ndim}'
    assert len(set([tensor.shape[0] for tensor in tensors])) == 1, \
        f'Expected the channels of all tensors must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in tensors]}'

    tensor_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    max_size = np.stack(tensor_sizes).max(0)

    if pad_size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (
                           max_size +
                           (pad_size_divisor - 1)) // pad_size_divisor * pad_size_divisor

    padded_samples = []
    for tensor in tensors:
        padding_size = [
            0, max_size[-1] - tensor.shape[-1], 0,
               max_size[-2] - tensor.shape[-2]
        ]
        if sum(padding_size) == 0:
            padded_samples.append(tensor)
        else:
            padded_samples.append(F.pad(tensor, padding_size, value=pad_value))

    return torch.stack(padded_samples, dim=0)


def tracking_samplelist_boxtype2tensor(batch_track_samples: TrackSampleList) -> None:
    for track_data_sample in batch_track_samples:
        for data_samples in track_data_sample.video_data_samples:
            if 'gt_instances' in data_samples:
                bboxes = data_samples.gt_instances.get('bboxes', None)
                if isinstance(bboxes, BaseBoxes):
                    data_samples.gt_instances.bboxes = bboxes.tensor
            if 'pred_instances' in data_samples:
                bboxes = data_samples.pred_instances.get('bboxes', None)
                if isinstance(bboxes, BaseBoxes):
                    data_samples.pred_instances.bboxes = bboxes.tensor
            if 'ignored_instances' in data_samples:
                bboxes = data_samples.ignored_instances.get('bboxes', None)
                if isinstance(bboxes, BaseBoxes):
                    data_samples.ignored_instances.bboxes = bboxes.tensor


class BoxInstMaskGenerator(BaseDataPreprocessor):

    def __init__(self,
                 *arg,
                 mask_stride: int = 4,
                 pairwise_size: int = 3,
                 pairwise_dilation: int = 2,
                 pairwise_color_thresh: float = 0.3,
                 bottom_pixels_removed: int = 10,
                 **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.mask_stride = mask_stride
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = pairwise_dilation
        self.pairwise_color_thresh = pairwise_color_thresh
        self.bottom_pixels_removed = bottom_pixels_removed

        if skimage is None:
            raise RuntimeError('skimage is not installed,\
                 please install it by: pip install scikit-image')

    def get_images_color_similarity(self, inputs: Tensor,
                                    image_masks: Tensor) -> Tensor:
        """Compute the image color similarity in LAB color space."""
        assert inputs.dim() == 4
        assert inputs.size(0) == 1

        unfolded_images = unfold_wo_center(
            inputs,
            kernel_size=self.pairwise_size,
            dilation=self.pairwise_dilation)
        diff = inputs[:, :, None] - unfolded_images
        similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

        unfolded_weights = unfold_wo_center(
            image_masks[None, None],
            kernel_size=self.pairwise_size,
            dilation=self.pairwise_dilation)
        unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

        return similarity * unfolded_weights

    def forward(self, data: dict, mean: None, std: None) -> dict:
        """Get pseudo mask labels using color similarity."""
        det_data = data
        inputs, data_samples = det_data['inputs'], det_data['data_samples']
        # get image masks and remove bottom pixels
        b_img_h, b_img_w = data_samples[0].batch_input_shape
        img_masks = []
        for i in range(inputs.shape[0]):
            img_h, img_w = data_samples[i].img_shape
            img_mask = inputs.new_ones((img_h, img_w))
            pixels_removed = int(self.bottom_pixels_removed *
                                 float(img_h) / float(b_img_h))
            if pixels_removed > 0:
                img_mask[-pixels_removed:, :] = 0
            pad_w = b_img_w - img_w
            pad_h = b_img_h - img_h
            img_mask = F.pad(img_mask, (0, pad_w, 0, pad_h), 'constant',
                             0.)
            img_masks.append(img_mask)
        img_masks = torch.stack(img_masks, dim=0)
        start = int(self.mask_stride // 2)
        img_masks = img_masks[:, start::self.mask_stride, start::self.mask_stride]

        # Get origin rgb image for color similarity
        ori_imgs = inputs * std + mean
        downsampled_imgs = F.avg_pool2d(
            ori_imgs.float(),
            kernel_size=self.mask_stride,
            stride=self.mask_stride,
            padding=0)

        # Compute color similarity for pseudo mask generation
        for im_i, data_sample in enumerate(data_samples):
            # TODO: Support rgb2lab in mmengine?
            images_lab = skimage.color.rgb2lab(
                downsampled_imgs[im_i].byte().permute(1, 2,
                                                      0).cpu().numpy())
            images_lab = torch.as_tensor(
                images_lab, device=ori_imgs.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = self.get_images_color_similarity(
                images_lab, img_masks[im_i])
            pairwise_mask = (images_color_similarity >=
                             self.pairwise_color_thresh).float()

            per_im_bboxes = data_sample.gt_instances.bboxes
            if per_im_bboxes.shape[0] > 0:
                per_im_masks = []
                for per_box in per_im_bboxes:
                    mask_full = torch.zeros((b_img_h, b_img_w),
                                            device=self.device).float()
                    mask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0
                    per_im_masks.append(mask_full)
                per_im_masks = torch.stack(per_im_masks, dim=0)
                pairwise_masks = torch.cat(
                    [pairwise_mask for _ in range(per_im_bboxes.shape[0])],
                    dim=0)
            else:
                per_im_masks = torch.zeros((0, b_img_h, b_img_w))
                pairwise_masks = torch.zeros(
                    (0, self.pairwise_size ** 2 - 1, b_img_h // self.mask_stride, b_img_w // self.mask_stride),
                    device=self.device)

            # TODO: Support BitmapMasks with tensor?
            start = int(self.mask_stride // 2)
            per_im_masks = per_im_masks[:, start::self.mask_stride, start::self.mask_stride]
            per_im_masks = per_im_masks.gt(0.5).float()
            data_sample.gt_instances.masks = BitmapMasks(
                per_im_masks.cpu().numpy(), b_img_h // self.mask_stride, b_img_w // self.mask_stride)
            data_sample.gt_instances.pairwise_masks = pairwise_masks
        return {'inputs': inputs, 'data_samples': data_samples}
