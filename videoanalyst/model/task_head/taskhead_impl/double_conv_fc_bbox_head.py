import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import ConvModule
from mmcv.cnn.utils.weight_init import normal_init, xavier_init
from mmcv.runner import BaseModule, ModuleList
from mmdet.models.backbones.resnet import Bottleneck

from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_fm_center_torch(score_size, score_offset, total_stride, device='cpu'):
    fm_height = score_size
    fm_width = score_size
    y = torch.arange(0, fm_height, dtype=torch.float, requires_grad=False, device=device)
    x = torch.arange(0, fm_width, dtype=torch.float, requires_grad=False, device=device)
    yy, xx = torch.meshgrid(y, x)

    xx = xx.unsqueeze(0).unsqueeze(-1)
    yy = yy.unsqueeze(0).unsqueeze(-1)
    fm_center = score_offset + torch.cat([xx, yy], dim=-1) * total_stride
    fm_center = fm_center.reshape(1, -1, 2)

    return fm_center


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


class BasicResBlock(BaseModule):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BasicResBlock, self).__init__(init_cfg)

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@TRACK_HEADS.register
class DoubleConvFCBBoxHead(ModuleBase):
    default_hyper_params = dict(
        in_channels=512,
        num_convs=0,
        input_size_adapt=False,
        x_size=0,
        score_size=0,
        total_stride=0,
    )

    def __init__(self):
        super(DoubleConvFCBBoxHead, self).__init__()
        self.bi_fc = torch.nn.Parameter(torch.tensor(0.))
        self.si_fc = torch.nn.Parameter(torch.tensor(1.))
        self.bi_conv = torch.nn.Parameter(torch.tensor(0.))
        self.si_conv = torch.nn.Parameter(torch.tensor(1.))

    def update_params(self):
        super(DoubleConvFCBBoxHead, self).update_params()

        self.in_channels = self._hyper_params['in_channels']
        self.num_convs = self._hyper_params['num_convs']
        self.total_stride = self._hyper_params["total_stride"]
        self.input_size_adapt = self._hyper_params["input_size_adapt"]

        x_size = self._hyper_params["x_size"]
        self.score_size = self._hyper_params["score_size"]
        score_offset = (x_size - 1 - (self.score_size - 1) * self.total_stride) // 2
        self._hyper_params["score_offset"] = score_offset
        self.score_offset = self._hyper_params["score_offset"]

        fm_ctr = get_fm_center_torch(self.score_size, self.score_offset, self.total_stride)
        self.register_buffer('fm_ctr', fm_ctr.clone().detach().requires_grad_(False))

        self.create_network()
        self.init_weights()

    def create_network(self):
        mid_channels = 1024

        # fc is implemented by 1x1 conv
        self.fc_branch = nn.Sequential(
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc_branch_cls = ConvModule(in_channels=mid_channels,
                                          out_channels=1,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)
        self.fc_branch_reg = ConvModule(in_channels=mid_channels,
                                          out_channels=4,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)

        convs = []
        for _ in range(self.num_convs):
            convs.append(Bottleneck(inplanes=mid_channels,
                                    planes=mid_channels // 4,
                                    conv_cfg=None,
                                    norm_cfg=dict(type='BN')))

        self.conv_branch = nn.Sequential(
            BasicResBlock(self.in_channels, mid_channels),
            *convs,
        )
        self.conv_branch_cls = ConvModule(in_channels=mid_channels,
                                          out_channels=1,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)
        self.conv_branch_reg = ConvModule(in_channels=mid_channels,
                                          out_channels=4,
                                          kernel_size=1,
                                          norm_cfg=dict(type='BN'),
                                          act_cfg=None)

    def init_weights(self):
        for m in self.conv_branch.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.fc_branch:
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # normal_init(self.fc_branch_cls, std=0.01)
        # normal_init(self.conv_branch_cls, std=0.01)
        # normal_init(self.fc_branch_reg, std=0.001)
        # normal_init(self.conv_branch_reg, std=0.001)

    def forward(self, x, x_size=0):
        output_fc_branch = self.fc_branch(x)
        cls_fc = self.fc_branch_cls(output_fc_branch)
        reg_fc = self.fc_branch_reg(output_fc_branch)

        cls_fc = cls_fc.permute(0, 2, 3, 1).reshape(cls_fc.shape[0], -1, 1)

        output_conv_branch = self.conv_branch(x)
        cls_conv = self.conv_branch_cls(output_conv_branch)
        reg_conv = self.conv_branch_reg(output_conv_branch)

        cls_conv = cls_conv.permute(0, 2, 3, 1).reshape(cls_conv.shape[0], -1, 1)

        reg_fc = torch.exp(self.si_fc * reg_fc + self.bi_fc) * self.total_stride
        bbox_fc = self.offset2bbox(reg_fc, x_size)
        reg_conv = torch.exp(self.si_conv * reg_conv + self.bi_conv) * self.total_stride
        bbox_conv = self.offset2bbox(reg_conv, x_size)

        return cls_fc, bbox_fc, cls_conv, bbox_conv

    def offset2bbox(self, offsets, x_size):
        # bbox decoding
        if self.input_size_adapt and x_size > 0:
            score_offsets = (x_size - 1 - (offsets.size(-1) - 1) * self.total_stride) // 2
            # fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset, self.total_stride)
            # fm_ctr = fm_ctr.to(offsets.device)
            fm_ctr = get_fm_center_torch(offsets.shape[-1], score_offsets, self.total_stride, offsets.device)
        else:
            fm_ctr = self.fm_ctr
        bbox = get_box(fm_ctr, offsets)
        return bbox

