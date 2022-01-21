import torch
import numpy as np
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
import torch.nn as nn
import math


class Sp_GCN(torch.nn.Module):
    def __init__(self, args, activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i == 0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                u.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                u.reset_param(w_i)
            self.w_list.append(w_i)

    def forward(self, A_list, Nodes_list, nodes_mask_list):
        node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        # take only last adj matrix in time
        Ahat = A_list[-1]
        # Ahat: NxN ~ 30k
        # sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l


class Sp_Skip_GCN(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.W_feat = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))

    def forward(self, A_list, Nodes_list=None):
        node_feats = Nodes_list[-1]
        # A_list: T, each element sparse tensor
        # take only last adj matrix in time
        Ahat = A_list[-1]
        # Ahat: NxN ~ 30k
        # sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(
            Ahat.matmul(l1.matmul(self.W2)) + (node_feats.matmul(self.W3))
        )

        return l2


class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)

    def forward(self, A_list, Nodes_list=None):
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        skip_last_l = torch.cat(
            (last_l, node_feats), dim=1
        )  # use node_feats.to_dense() if 2hot encoded input
        return skip_last_l


class Sp_GCN_LSTM_A(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.rnn = nn.LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )

    def forward(self, A_list, Nodes_list=None, nodes_mask_list=None):
        last_l_seq = []
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            # A_list: T, each element sparse tensor
            # note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]


class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.rnn = nn.GRU(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )


class Sp_GCN_LSTM_B(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        assert args.num_layers == 2, "GCN-LSTM and GCN-GRU requires 2 conv layers."
        self.rnn_l1 = nn.LSTM(
            input_size=args.layer_1_feats,
            hidden_size=args.lstm_l1_feats,
            num_layers=args.lstm_l1_layers,
        )

        self.rnn_l2 = nn.LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats))
        u.reset_param(self.W2)

    def forward(self, A_list, Nodes_list=None, nodes_mask_list=None):
        l1_seq = []
        l2_seq = []
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            # A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out[-1]


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    def __init__(self, args, activation):
        super().__init__(args, activation)
        self.rnn_l1 = nn.GRU(
            input_size=args.layer_1_feats,
            hidden_size=args.lstm_l1_feats,
            num_layers=args.lstm_l1_layers,
        )

        self.rnn_l2 = nn.GRU(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_l2_feats,
            num_layers=args.lstm_l2_layers,
        )


class Random(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

    def forward(self, edge_index, node_feats, edge_feats, nodes_mask_list):
        num_nodes = node_feats[0].size()[0]
        return torch.rand(num_nodes, self.args.layer_2_feats).to(self.device)


class Classifier(torch.nn.Module):
    def __init__(self, args, out_features, in_features):
        super(Classifier, self).__init__()
        activation = torch.nn.ReLU()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features, out_features=args.gcn_parameters["cls_feats"]
            ),
            activation,
            torch.nn.Linear(
                in_features=args.gcn_parameters["cls_feats"], out_features=out_features
            ),
        )

    def forward(self, x):
        return self.mlp(x)


class TLP_Classifier(torch.nn.Module):
    def __init__(self, args, classifier, time_encoder):
        super(TLP_Classifier, self).__init__()
        # activation = torch.nn.ReLU()
        self.mlp = classifier
        self.time_encoder = time_encoder

    def forward(self, x, t):
        te = self.time_encoder(t)
        start_te = te[:, 0, :]
        duration_te = te[:, 1, :]
        xd = torch.cat([x, start_te, duration_te], dim=1)
        return self.mlp(xd)


# Borrowed from TGN
class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
            .float()
            .reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        # t = t.unsqueeze(dim=2) # We don't need this since we only have one time stamp.

        ## Consider doing something super simple. e.g.
        # inp = x.repeat(10,1).transpose(0,1)
        # div = torch.cat([torch.ones((x.shape[0],1))*10**(bits-1-i) for i in range(bits)],1)
        # return (((inp/div).int()%10)*0.1).float()

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output
