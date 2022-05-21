import math
import torch
import torch.nn as nn


class SpatialPositionEncodingSine(nn.Module):
    def __init__(self, d_model, score_size):
        super(SpatialPositionEncodingSine, self).__init__()
        self.position_encoding = self.init(d_model, score_size)

    def init(self, d_model, score_size):
        eps = 1e-6
        norm_scale = 2 * math.pi
        temperature = 10000
        num_pos_feats = d_model // 2

        ones = torch.ones(1, score_size, score_size)
        y_embed = ones.cumsum(1, dtype=torch.float32)
        x_embed = ones.cumsum(2, dtype=torch.float32)

        # normalize
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * norm_scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * norm_scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        pos = pos.unsqueeze(1)  # B, C, H, W
        return pos

    def forward(self, x):
        if len(x.shape) == 4:
            pos = self.position_encoding.to(x.device)
        elif len(x.shape) == 5:
            pos = self.position_encoding.to(x.device).unsqueeze(1)
        else:
            raise ValueError('The shape [{}] of input is invalid'.format(x.shape))
        return x + pos


class SpatialPositionEncodingLearned(nn.Module):
    def __init__(self, d_model, score_size):
        super(SpatialPositionEncodingLearned, self).__init__()
        self.row_embed = nn.Embedding(score_size, d_model // 2)
        self.col_embed = nn.Embedding(score_size, d_model // 2)
        self.spatial_size = score_size
        self.pos = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def get_position_encoding(self, x):
        i = torch.arange(self.spatial_size, device=x.device)
        j = torch.arange(self.spatial_size, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(self.spatial_size, 1, 1),
            y_emb.unsqueeze(1).repeat(1, self.spatial_size, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1, C, H, W
        return pos

    def forward(self, x):
        if self.training:
            self.pos = self.get_position_encoding(x)
        else:
            if self.pos is None:
                self.pos = self.get_position_encoding(x)
        return x + self.pos


class TemporalPositionEncoding(nn.Module):
    def __init__(self, d_model):
        super(TemporalPositionEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        ic = torch.arange(self.d_model, device=x.device).float()
        dem = torch.pow(10000, 2 * (ic // 2) / self.d_model).unsqueeze(0)
        it = torch.arange(x.shape[1], device=x.device).unsqueeze(1)
        sinusoid_table = it / dem
        sinusoid_table[:, 0::2] = sinusoid_table[:, 0::2].sin()
        sinusoid_table[:, 1::2] = sinusoid_table[:, 1::2].cos()
        sinusoid_table = sinusoid_table[None, :, None, None, :]  # B, T, 1, 1, C
        sinusoid_table = sinusoid_table.permute(0, 1, 4, 2, 3).contiguous()  # B, T, C, 1, 1
        return x + sinusoid_table


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # B, #heads, THW, (T)HW

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.modules():
            if isinstance(p, nn.Linear):
                n = p.in_features
                y = 1.0 / math.sqrt(n)
                if hasattr(p, "weight") and p.weight is not None:
                    p.weight.data.uniform_(-y, y)
                if hasattr(p, "bias") and p.bias is not None:
                    p.bias.data.fill_(0)

    def forward(self, q, k, v, mask=None):
        # q: B, THW, C
        # k: B, HW, C
        # k: B, HW, C
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        residual = q

        q = self.w_qs(q).view(*q.shape[:2], n_head, d_k).transpose(1, 2).contiguous()
        k = self.w_ks(k).view(*k.shape[:2], n_head, d_k).transpose(1, 2).contiguous()
        v = self.w_vs(v).view(*v.shape[:2], n_head, d_v).transpose(1, 2).contiguous()

        # Transpose for attention dot product: [B, #heads, THW, d_k/d_v]
        # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiguous().view(*residual.shape[:2], -1)
        out = self.dropout(self.fc(out))
        out += residual

        out = self.layer_norm(out)

        return out, attn


class PositionWiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.modules():
            if isinstance(p, nn.Linear):
                n = p.in_features
                y = 1.0 / math.sqrt(n)
                if hasattr(p, "weight") and p.weight is not None:
                    p.weight.data.uniform_(-y, y)
                if hasattr(p, "bias") and p.bias is not None:
                    p.bias.data.fill_(0)

    def forward(self, x):
        residual = x

        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
