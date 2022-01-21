import torch
import numpy as np
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
import torch.nn as nn
import math


class WSDM_Decoder(torch.nn.Module):
    def __init__(
        self,
        args,
        time_encoder,
        out_features,
        nodes_feats,
        in_feats_etype,
        out_feats_etype,
        in_feats_time,
        out_feats_time,
    ):
        super(WSDM_Decoder, self).__init__()

        self.time_encoder = time_encoder
        self.activation = torch.nn.ReLU()

        self.etype_linear = torch.nn.Linear(
            in_features=in_feats_etype, out_features=out_feats_etype
        )
        self.time_linear = torch.nn.Linear(
            in_features=in_feats_time, out_features=out_feats_time
        )

        in_features = nodes_feats + out_feats_etype + out_feats_time
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features,
                out_features=(2 * args.gcn_parameters["cls_feats"]),
            ),
            self.activation,
            torch.nn.Linear(
                in_features=(2 * args.gcn_parameters["cls_feats"]),
                out_features=args.gcn_parameters["cls_feats"],
            ),
            self.activation,
            torch.nn.Linear(
                in_features=args.gcn_parameters["cls_feats"],
                out_features=args.gcn_parameters["cls_feats"],
            ),
            self.activation,
            torch.nn.Linear(
                in_features=args.gcn_parameters["cls_feats"], out_features=out_features
            ),
        )

    # Borrowed from WSDM cup baseline
    def time_encoding(self, x, bits=10):
        x = x.squeeze()
        inp = x.repeat(10, 1).transpose(0, 1)
        div = torch.cat(
            [torch.ones((x.shape[0], 1)) * 10 ** (bits - 1 - i) for i in range(bits)], 1
        )
        return (((inp / div).int() % 10) * 0.1).float()

    def forward(self, x, t, e_type, t_feats):
        emb_etype = self.etype_linear(e_type)
        # emb_etype = self.activation(emb_etype)

        emb_t = self.time_encoding(t)
        # emb_t = self.activation(emb_t)
        time_vec = torch.cat(
            [emb_t, t_feats], dim=1
        )  # t_feats needs work if multiple feats
        emb_time = self.time_linear(time_vec)
        # emb_time = self.activation(emb_time)

        # import pdb; pdb.set_trace()
        # print(x.size(), emb_etype.size(), emb_time.size())
        emb_edge = torch.cat([x, emb_etype, emb_time], dim=1)
        return self.mlp(emb_edge)
