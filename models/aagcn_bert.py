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

class AAGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(AAGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc4 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc5 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc6 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc7 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc8 = GraphConvolution(opt.bert_dim, opt.bert_dim)


        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.fc2 = nn.Linear(4 * opt.hidden_dim, opt.polarities_dim)
        self.dfc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self, inputs):

        text_bert_indices, bert_segments_ids, e_adj, a_adj = inputs
        encoder_layer, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        text_out = encoder_layer
        x = F.relu(self.gc1(text_out, e_adj))
        x = F.relu(self.gc2(x, a_adj))

        x = F.relu(self.gc3(x, e_adj))
        x = F.relu(self.gc4(x, a_adj))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)

        output = self.fc(x)
        return output
