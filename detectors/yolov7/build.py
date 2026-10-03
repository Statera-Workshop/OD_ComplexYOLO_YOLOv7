"""
YOLOv7 builders for the three tasks.

Source: Darknet yolov7.cfg by the paper authors (AlexeyAB/darknet, cfg/yolov7.cfg),
        paper: Wang, Bochkovskiy, Liao, "YOLOv7: Trainable bag-of-freebies ...", 2022.

Architecture (algorithm-specific, this folder):
    E-ELAN backbone (SiLU), SPPCSPC, ELAN PAN neck, three scales (strides 8, 16, 32).
    The cfg is the deploy form: RepConv branches are already merged into plain 3x3 convs.

Detection layers: reused unchanged from detectors/yolov3, exactly as v2 / v4 / v5.
    image   -> Image YOLO layer   (anchors of the original yolov7.cfg)
    complex -> Complex-YOLO layer (E-RPN im/re)   anchors: v3 / v4 KITTI BEV anchors
    yolo3d  -> YOLO3D layer       (direct yaw, z, h)

Delta vs v3 / v4: backbone + neck only.

Not used from the paper (same policy as v4 / v5):
    - auxiliary head + coarse-to-fine lead-guided label assignment
    - training-time re-parameterised RepConv (cfg already merged)
    - scale_x_y = 2 grid sensitivity, CIoU loss  -> shared YOLO layers
    - Mosaic and other training tricks            -> common protocol
"""

from pathlib import Path

from common.darknet import Darknet
from detectors.yolov3.build import complex_factory

CFG_DIR = Path(__file__).resolve().parent / "cfg"

SPECS = {
    "complex": ("complex_yolov7.cfg", complex_factory, 7),
}


def build_model(task, num_classes):
    cfg_name, factory, num_box_attr = SPECS[task]
    return Darknet(CFG_DIR / cfg_name, factory, num_classes, num_box_attr)
