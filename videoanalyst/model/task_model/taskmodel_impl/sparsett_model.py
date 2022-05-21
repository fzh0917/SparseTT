# -*- coding: utf-8 -*

from loguru import logger

import torch

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(
        pretrain_model_path="",
        in_channels=768,
        mid_channels=512,
        conv_weight_std=0.01,
        corr_feat_output=False,
        amp=False
    )

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, neck, head, loss=None):
        super(SiamTrack, self).__init__()
        self.basemodel = backbone
        self.neck = neck
        self.head = head
        self.loss = loss
        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def train_forward(self, training_data):
        target_img = training_data["im_z"]
        search_img = training_data["im_x"]
        # backbone feature
        f_z = self.basemodel(target_img)
        f_x = self.basemodel(search_img)
        # feature adjustment
        f_z = self.feat_adjuster_z(f_z)
        f_x = self.feat_adjuster_x(f_x)
        # feature matching
        output = self.neck(f_x, f_z)
        # head
        cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(output)
        predict_data = dict(
            cls_fc=cls_fc,
            bbox_fc=bbox_fc,
            cls_conv=cls_conv,
            bbox_conv=bbox_conv
        )
        if self._hyper_params["corr_feat_output"]:
            predict_data["corr_feat"] = output
        return predict_data

    def test_forward(self, f_x, enc_output, x_size):
        # feature matching
        output = self.neck.decode(f_x, enc_output)
        # head
        cls_fc, bbox_fc, cls_conv, bbox_conv = self.head(output, x_size)
        # apply sigmoid
        cls_fc = torch.sigmoid(cls_fc)
        cls_conv = torch.sigmoid(cls_conv)
        # merge two cls socres
        cls_score_final = cls_fc + cls_conv * (1 - cls_fc)
        # register extra output
        extra = dict()  # for faster inference
        # extra = {"f_x": f_x, "encoder_output": enc_output, "decoder_output": output}
        # output
        out_list = cls_score_final, bbox_conv, extra
        return out_list

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            target_img, = args
            # backbone feature
            f_z = self.basemodel(target_img)
            # template as kernel
            f_z = self.feat_adjuster_z(f_z)
            enc_output = self.neck.encode(f_z)
            # output
            out_list = [enc_output]
        elif phase == 'track':
            assert len(args) == 2, "Illegal args length: %d" % len(args)
            search_img, enc_output = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            f_x = self.feat_adjuster_x(f_x)
            out_list = self.test_forward(f_x, enc_output, search_img.size(-1))
            out_list[-1]
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        # self._initialize_conv()
        super().update_params()

    def _make_convs(self):
        in_channels = self._hyper_params['in_channels']
        mid_channels = self._hyper_params['mid_channels']

        # feature adjustment
        self.feat_adjuster_z = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)
        self.feat_adjuster_x = conv_bn_relu(in_channels, mid_channels, kszie=1, has_relu=False, bn_eps=1e-3)
        # self.feat_adjuster_z = torch.nn.Identity()
        # self.feat_adjuster_x = torch.nn.Identity()

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [self.feat_adjuster_z.conv, self.feat_adjuster_x.conv]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight, std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
