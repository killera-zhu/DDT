import torch.nn as nn
import torch
from math import sqrt

import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import math
import torch.fft
from einops import rearrange


class EncoderLayer(nn.Module):
    def __init__(
            self,
            attention,
            d_model,
            d_ff=None,
            dropout=0.1,
            activation="relu",
            mask_generator=None
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.mask_generator = mask_generator  # 动态掩码生成器

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 生成动态掩码
        dynamic_mask = None
        if self.mask_generator is not None:
            dynamic_mask = self.mask_generator(x)  # 形状: (B, 1, L, L)

        # 生成因果掩码
        B, L, _ = x.shape
        causal_mask = torch.tril(torch.ones((B, 1, L, L), device=x.device)).bool()

        # 合并掩码
        if dynamic_mask is not None:
            if attn_mask is not None:
                attn_mask = attn_mask * dynamic_mask * causal_mask
            else:
                attn_mask = dynamic_mask * causal_mask
        elif attn_mask is not None:
            attn_mask = attn_mask * causal_mask
        else:
            attn_mask = causal_mask

        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)
            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Mahalanobis_mask(nn.Module):
    def __init__(self):
        super(Mahalanobis_mask, self).__init__()
        # 移除了静态的 input_size 依赖

    def forward(self, X):
        p = self.calculate_prob_distance(X)
        sample = self.bernoulli_gumbel_rsample(p)
        mask = sample.unsqueeze(1)  # 形状: (B, 1, L, L)
        return mask

    def calculate_prob_distance(self, X):
        # 动态计算特征维度 D
        D = X.size(-1)
        frequency_size = D // 2 + 1

        # 动态生成可训练矩阵 A（每次输入维度变化时重新初始化）
        A = nn.Parameter(torch.randn(frequency_size, frequency_size, device=X.device), requires_grad=True)

        XF = torch.abs(torch.fft.rfft(X, dim=-1))  # [B, C, D']
        X1 = XF.unsqueeze(2)  # [B, C, 1, D']
        X2 = XF.unsqueeze(1)  # [B, 1, C, D']
        diff = X1 - X2  # [B, C, C, D']

        # 修正 einsum 下标对齐
        temp = torch.einsum("dk,bijd->bijd", A, diff)  # [B, C, C, D']
        dist = torch.einsum("bijd,bijd->bij", temp, temp)  # [B, C, C]

        # 后续逻辑保持不变
        exp_dist = 1 / (dist + 1e-10)
        identity_matrices = 1 - torch.eye(exp_dist.shape[-1], device=X.device)
        exp_dist = exp_dist * identity_matrices.unsqueeze(0)  # 屏蔽对角线
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        p = exp_dist / (exp_max + 1e-10)
        p = p * 0.99 + (torch.eye(p.shape[-1], device=X.device)) * 0.01  # 保持数值稳定
        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix