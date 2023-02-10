import numpy as np
import scipy.sparse as sp
import sys
import torch
from torch import nn, optim
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.io as scio
import os
import time

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)  
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input_features, adj):  
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    
class HCNH(nn.Module):

    """
    Hypergraph Convolution on Nodes-Hyperedges (HCNH), https://dl.acm.org/doi/abs/10.1145/3494567
    """

    def __init__(self, input_feat_x_dim, input_feat_y_dim, latent_dim, nclass):
        super(HCNH, self).__init__()
        
        self.gcx1 = GraphConvolution(input_feat_x_dim, latent_dim)
        self.gcx2 = GraphConvolution(latent_dim, nclass)

        self.gcy1 = GraphConvolution(input_feat_y_dim, latent_dim)
        self.gcy2 = GraphConvolution(latent_dim, nclass)
            

    def forward(self, x, hx, y, hy):
        
        neg_slope = 0.2
        
        """ filtering on nodes """
        x = self.gcx1(x, hx)
        x = F.leaky_relu(x, negative_slope=neg_slope)        
 
        x = self.gcx2(x, hx)
        x = F.leaky_relu(x, negative_slope=neg_slope)

        
        """ filtering on hyperedges """
        y = self.gcy1(y, hy)
        y = F.leaky_relu(y, negative_slope=neg_slope)        
        
        y = self.gcy2(y, hy)
        y = F.leaky_relu(y, negative_slope=neg_slope)
        
        
        """ recover hypergraph """
        h = torch.mm(x, y.t())
        h = torch.sigmoid(h)
        
        output = F.log_softmax(x, dim=1)
        
        return h, output, x, y
