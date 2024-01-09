
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


from mmdet.models.detectors import SingleStageDetector
from .mask2former_vid import Mask2formerVideo

@MODELS.register_module()
class YOSOVideoSam(Mask2formerVideo):
    OVERLAPPING = None

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 inference_sam: bool = False,
                 init_cfg: OptMultiConfig = None
                 ):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.alpha = 0.4
        self.beta = 0.8

        self.inference_sam = inference_sam
