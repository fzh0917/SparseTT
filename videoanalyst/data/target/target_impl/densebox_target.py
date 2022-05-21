# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict
from mmcv import imnormalize

from ..target_base import TRACK_TARGETS, TargetBase
from .utils import make_densebox_target


@TRACK_TARGETS.register
class DenseboxTarget(TargetBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        z_size=127,
        x_size=303,
        score_size=17,
        score_offset=87,
        total_stride=8,
        normalize=False,
        norm_mean=[123.675, 116.28, 103.53],
        norm_std=[58.395, 57.12, 57.375],
        to_rgb=False,
    )

    def __init__(self) -> None:
        super().__init__()

    def update_params(self):
        hps = self._hyper_params
        # hps['score_size'] = (
        #     hps['x_size'] -
        #     hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self.normalize = hps['normalize']
        self.norm_mean = np.array(hps['norm_mean'], dtype=np.float32)
        self.norm_std = np.array(hps['norm_std'], dtype=np.float32)
        self.to_rgb = hps['to_rgb']
        self._hyper_params = hps

    def __call__(self, sampled_data: Dict) -> Dict:
        data_z = sampled_data["data1"]
        im_z, bbox_z = data_z["image"], data_z["anno"]

        data_x = sampled_data["data2"]
        im_x, bbox_x = data_x["image"], data_x["anno"]

        is_negative_pair = sampled_data["is_negative_pair"]

        if self.normalize:
            im_z = imnormalize(im_z, self.norm_mean, self.norm_std, self.to_rgb)
            im_x = imnormalize(im_x, self.norm_mean, self.norm_std, self.to_rgb)

        # input tensor
        im_z = im_z.transpose(2, 0, 1)
        im_x = im_x.transpose(2, 0, 1)

        # training target
        cls_label, ctr_label, box_label = make_densebox_target(
            bbox_x.reshape(1, 4), self._hyper_params)
        if is_negative_pair:
            cls_label[cls_label == 0] = -1
            cls_label[cls_label == 1] = 0

        training_data = dict(
            im_z=im_z,
            im_x=im_x,
            bbox_z=bbox_z,
            bbox_x=bbox_x,
            cls_gt=cls_label,
            ctr_gt=ctr_label,
            box_gt=box_label,
            is_negative_pair=int(is_negative_pair),
        )
        #training_data = super().__call__(training_data)

        return training_data
