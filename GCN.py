# import torchvision.models as models
# from util import *
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RelationNet(nn.Module):

    def __init__(self, num_classes, task, in_channel=200, out_channel=100,  adj_file=None):
        super(RelationNet, self).__init__()
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, out_channel)
        self.relu = nn.LeakyReLU(0.2)

        data_adj, knowledge_adj = pickle.load(open(adj_file, 'rb'))
        self.A_data = torch.from_numpy(data_adj).float().cuda()
        self.A_know = torch.from_numpy(knowledge_adj).float().cuda()

        self.t = 10.0
        self.W = nn.Parameter(torch.FloatTensor(in_channel, 200))
        init.xavier_uniform_(self.W)


    def gen_adj(self, A, t=0):
        A[A < t] = 0
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj


    def forward(self, feature, inp):
        '''
        :param feature:
        :param inp:
        :return:
        '''
        data_adj = self.gen_adj(self.A_data, self.t)
        know_adj = self.gen_adj(self.A_know)
        adj = torch.stack([data_adj, know_adj], dim=0) # (2, 22, 22)

        x = self.gc1(inp, adj)  # (2, 22, 1024)
        x = self.relu(x)
        x = self.gc2(x, adj)  # (2, 22, 100)

        x = x.transpose(1, 2).contiguous().view(-1, self.num_classes)  # (3, 100, 22 or 20)
        # x = x.transpose(0, 1)
        if not self.training:
            pickle.dump(x, open("./label_embedding.pkl", 'wb'))

        x = torch.einsum("bk,kd,dc->bc", feature, self.W, x)
        # x = torch.matmul(feature, x)
        return x

