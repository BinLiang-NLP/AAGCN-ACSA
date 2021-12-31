# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class AAGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AAGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc5 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc6 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)

        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.fc2 = nn.Linear(4 * opt.hidden_dim, opt.polarities_dim)
        self.dfc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)


    def forward(self, inputs):
        text_indices, adj, d_adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1).cpu()
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)

        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, d_adj))

        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, d_adj))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)

        output = self.fc(x)
        return output
