import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_base import TRACK_NECKS


@TRACK_NECKS.register
class TransformerNeck(ModuleBase):
    default_hyper_params = dict(
        mid_channels_model=256,
        mid_channels_ffn=2048,
        num_heads=8,
        num_encoder_layers=8,
        num_decoder_layers=8,
        prob_dropout=0.0,
        f_z_size=5,
        f_x_size=25,
        top_k=None
    )

    def __init__(self):
        super(TransformerNeck, self).__init__()

    def update_params(self):
        super().update_params()
        mid_channels_model = self._hyper_params['mid_channels_model']
        mid_channels_ffn = self._hyper_params['mid_channels_ffn']
        num_heads = self._hyper_params['num_heads']
        num_encoder_layers = self._hyper_params['num_encoder_layers']
        num_decoder_layers = self._hyper_params['num_decoder_layers']
        prob_dropout = self._hyper_params['prob_dropout']
        f_z_size = self._hyper_params['f_z_size']
        f_x_size = self._hyper_params['f_x_size']
        top_k = self._hyper_params['top_k']
        self.encoder = Encoder(mid_channels_model=mid_channels_model,
                               mid_channels_ffn=mid_channels_ffn,
                               num_heads=num_heads,
                               num_layers=num_encoder_layers,
                               prob_dropout=prob_dropout,
                               score_size=f_z_size)
        self.decoder = Decoder(mid_channels_model=mid_channels_model,
                               mid_channels_ffn=mid_channels_ffn,
                               num_heads=num_heads,
                               num_layers=num_decoder_layers,
                               prob_dropout=prob_dropout,
                               score_size=f_x_size,
                               top_k=top_k)

    def encode(self, f_z):
        return self.encoder(f_z)

    def decode(self, f_x, enc_output):
        dec_output = self.decoder(f_x, enc_output)
        final_output = torch.cat([dec_output, f_x], dim=1)
        return final_output

    def forward(self, f_x, f_z):
        enc_output = self.encode(f_z)
        final_output = self.decode(f_x, enc_output)
        return final_output
