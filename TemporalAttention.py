import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import ipdb

from common_layer import PositionalEncoding, _gen_timing_signal, PositionwiseFeedForward


class TemporalAttention(torch.nn.Module):
    def __init__(self, d_model, K, d):
        super(TemporalAttention, self).__init__()
        # self.d_model = d_model
        self.d = d
        self.K = K
        self.D = K*d

        self.fc_q = nn.Linear(self.D,self.D)
        self.fc_k = nn.Linear(self.D,self.D)
        self.fc_v = nn.Linear(self.D,self.D)
        self.fc_1 = nn.Linear(self.D,self.D)
        # self.fc_2 = nn.Linear(self.D,self.d_model)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.layer_norm_mha = LayerNorm(self.D)
        self.layer_norm_ffn = LayerNorm(self.D)

        self.dropout = nn.Dropout(0.1)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.D, 2 * self.D),
            nn.ReLU(),
            nn.Linear(2 * self.D, self.D),
        )

    def forward(self, X):
        # ipdb.set_trace()
        #B*W*H,T,48
        N,T,D = X.shape
        PE = _gen_timing_signal(T,D)
        # ipdb.set_trace()
        X = X+PE[:,:X.shape[1],:].type_as(X.data)
        # X_res = X
        # Layer Normalization
        X_norm = self.layer_norm_mha(X)
        X = X_norm
        # multihead attention
        query = F.relu(self.fc_q(X))
        key = F.relu(self.fc_k(X))
        value = F.relu(self.fc_v(X))
        query = torch.cat(torch.split(query,self.d,dim=-1),dim=0)
        key = torch.cat(torch.split(key,self.d,dim=-1),dim=0)
        value = torch.cat(torch.split(value,self.d,dim=-1),dim=0)
        key = torch.transpose(key,1,2)
        # value = torch.transpose(value,1,2)
        att = torch.matmul(query,key)
        att /= (self.d ** 0.5)
        att = self.softmax(att)
        out = torch.matmul(att, value)
        out = torch.cat(torch.split(out, out.shape[0]//self.K, dim=0), dim=-1)
        # out = self.fc_2(F.relu(self.fc_1(out))) # seems from GMAN
        out = self.fc_1(out)

        # # Dropout and residual after self-attention
        X = self.dropout(X + out)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(X)

        # Positionwise Feedforward
        y = self.feed_forward(x_norm)
        # Dropout and residual
        y = self.dropout(X + y)

        return y


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta