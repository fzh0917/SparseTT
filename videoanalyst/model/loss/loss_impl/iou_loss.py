# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn as nn

from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES
from .utils import SafeLog

eps = np.finfo(np.float32).tiny


class IOULoss(nn.Module):

    # default_hyper_params = dict(
    #     name="iou_loss",
    #     background=0,
    #     ignore_label=-1,
    #     weight=1.0,
    # )

    def __init__(self,
                 background=0,
                 ignore_label=-1,
                 weight=1.0,
                 branch_name=""):
        super().__init__()
        self.safelog = SafeLog()
        self.background = background
        self.ignore_label = ignore_label
        self.weight = weight
        self.iou_key = "iou_{}".format(branch_name)
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.register_buffer("t_zero", torch.tensor(0., requires_grad=False))

    def forward(self, pred, gt, cls_gt):
        # pred = pred_data[self.name]
        # gt = target_data["box_gt"]
        # cls_gt = target_data["cls_gt"]
        mask = ((~(cls_gt == self.background)) *
                (~(cls_gt == self.ignore_label))).detach()
        mask = mask.type(torch.Tensor).squeeze(2).to(pred.device)

        aog = torch.abs(gt[:, :, 2] - gt[:, :, 0] +
                        1) * torch.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
        aop = torch.abs(pred[:, :, 2] - pred[:, :, 0] +
                        1) * torch.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

        iw = torch.min(pred[:, :, 2], gt[:, :, 2]) - torch.max(
            pred[:, :, 0], gt[:, :, 0]) + 1
        ih = torch.min(pred[:, :, 3], gt[:, :, 3]) - torch.max(
            pred[:, :, 1], gt[:, :, 1]) + 1
        inter = torch.max(iw, self.t_zero) * torch.max(ih, self.t_zero)

        union = aog + aop - inter
        iou = torch.max(inter / union, self.t_zero)
        loss = -self.safelog(iou)

        loss = (loss * mask).sum() / torch.max(mask.sum(), self.t_one) * self.weight
        iou = iou.detach()
        iou = (iou * mask).sum() / torch.max(mask.sum(), self.t_one)
        extra = {self.iou_key: iou}

        return loss, extra


@TRACK_LOSSES.register
class IOULossFC(ModuleBase):
    default_hyper_params = dict(
        name="iou_loss",
        weight=1.0,
    )

    def __init__(self):
        super(IOULossFC, self).__init__()

    def update_params(self):
        super(IOULossFC, self).update_params()
        self.name = self._hyper_params['name']
        self.criterion = IOULoss(weight=self._hyper_params['weight'], branch_name="fc")

    def forward(self, pred_data, target_data):
        bbox_fc = pred_data['bbox_fc']
        bbox_gt = target_data['box_gt']
        cls_gt = target_data['cls_gt']
        return self.criterion(bbox_fc, bbox_gt, cls_gt)


@TRACK_LOSSES.register
class IOULossConv(ModuleBase):
    default_hyper_params = dict(
        name="iou_loss",
        weight=1.0,
    )

    def __init__(self):
        super(IOULossConv, self).__init__()

    def update_params(self):
        super(IOULossConv, self).update_params()
        self.name = self._hyper_params['name']
        self.criterion = IOULoss(weight=self._hyper_params['weight'], branch_name="conv")

    def forward(self, pred_data, target_data):
        bbox_conv = pred_data['bbox_conv']
        bbox_gt = target_data['box_gt']
        cls_gt = target_data['cls_gt']
        return self.criterion(bbox_conv, bbox_gt, cls_gt)


if __name__ == '__main__':
    B = 16
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.int8)
    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    criterion_cls = SigmoidCrossEntropyRetina()
    loss_cls = criterion_cls(pred_cls, gt_cls)

    criterion_ctr = SigmoidCrossEntropyCenterness()
    loss_ctr = criterion_ctr(pred_ctr, gt_ctr, gt_cls)

    criterion_reg = IOULoss()
    loss_reg = criterion_reg(pred_reg, gt_reg, gt_cls)

    from IPython import embed
    embed()
