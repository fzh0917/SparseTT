import math
import torch
import torch.nn as nn

from videoanalyst.model.utils.transformer_layers import (SpatialPositionEncodingLearned,
                                                         MultiHeadAttention,
                                                         PositionWiseFeedForward)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output, enc_slf_attn = self.slf_attn(query=enc_input, key=enc_input, value=enc_input,
                                                 attn_mask=mask)
        enc_output = enc_input + enc_output
        enc_output = self.norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 mid_channels_ffn=2048,
                 num_heads=8,
                 num_layers=8,
                 prob_dropout=0.0,
                 score_size=33):
        super(Encoder, self).__init__()
        assert mid_channels_model % num_heads == 0
        mid_channels_k = mid_channels_model // num_heads
        mid_channels_v = mid_channels_k

        self.spatial_position_encoding = SpatialPositionEncodingLearned(mid_channels_model, score_size)
        # self.dropout = nn.Dropout(p=prob_dropout)
        # self.layer_norm = nn.LayerNorm(mid_channels_model, eps=1e-6)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(mid_channels_model, mid_channels_ffn, num_heads,
                         mid_channels_k, mid_channels_v, dropout=prob_dropout)
            for _ in range(num_layers)])

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.spatial_position_encoding(x)

        x = x.view(*x.shape[:2], -1)  # B, C, HW
        enc_output = x.permute(2, 0, 1).contiguous()  # HW, B, C

        # no need of mask if enc_output.shape is (BT, HW, C)
        for enc_layer in self.encoder_layers:
            enc_output, enc_slf_attn = enc_layer(enc_output)
        return enc_output  # HW, B, C
