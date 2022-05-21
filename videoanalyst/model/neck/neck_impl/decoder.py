import math
import torch
import torch.nn as nn

from videoanalyst.model.utils.multi_head_attention_topk import MultiHeadAttentionTopK
from videoanalyst.model.utils.transformer_layers import (SpatialPositionEncodingLearned,
                                                         TemporalPositionEncoding,
                                                         MultiHeadAttention,
                                                         PositionWiseFeedForward)


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, top_k=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttentionTopK(embed_dim=d_model, num_heads=n_head, dropout=dropout, top_k=top_k)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, tgt_mask=None, src_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=tgt_mask)
        dec_output = dec_input + dec_output
        dec_output = self.norm1(dec_output)

        dec_output2, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=src_mask)
        dec_output2 = dec_output + dec_output2
        dec_output2 = self.norm2(dec_output2)

        dec_output2 = self.pos_ffn(dec_output2)
        return dec_output2, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self,
                 mid_channels_model=256,
                 mid_channels_ffn=2048,
                 num_heads=8,
                 num_layers=8,
                 prob_dropout=0.0,
                 score_size=33,
                 top_k=None):
        super(Decoder, self).__init__()
        assert mid_channels_model % num_heads == 0
        mid_channels_k = mid_channels_model // num_heads
        mid_channels_v = mid_channels_k

        self.spatial_position_encoding = SpatialPositionEncodingLearned(mid_channels_model, score_size)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(mid_channels_model, mid_channels_ffn, num_heads,
                         mid_channels_k, mid_channels_v, top_k, dropout=prob_dropout)
            for _ in range(num_layers)])

    def forward(self, pre_output, enc_output):
        B, C, H, W = pre_output.shape
        pre_output = self.spatial_position_encoding(pre_output)
        pre_output = pre_output.view(B, C, -1)  # B, C, HW
        dec_output = pre_output.permute(2, 0, 1).contiguous()  # HW, B, C

        for dec_layer in self.decoder_layers:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output)

        dec_output = dec_output.permute(1, 2, 0).contiguous()  # B, C, HW
        dec_output = dec_output.view(*dec_output.shape[:2], H, W)

        return dec_output
