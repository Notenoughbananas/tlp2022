import torch
import numpy as np
import pandas as pd
import taskers_utils as tu
import utils as u
from functools import partial


class Tlp_Labeler:
    def __init__(self, args, continuous_dataset):

        self.args = args
        self.data_folder = "data"
        self.get_data = None  # A function that takes an idx and returns a label_adj
        self.timespan = args.query_horizon  # In number of snapshots looking ahead
        self.cdata = continuous_dataset
        self.dataset_name = self.cdata.name
        self.max_links = (
            args.max_label_links
        )  # Max number of total samples while training

        if self.dataset_name == "wsdm-A":
            left = 86400  # 1 day
            mode = 172800  # 2 days
            right = 259200  # 3 days
        elif self.dataset_name == "wsdm-B":
            left = 0  # 0 days
            mode = 86400  # 1 day
            right = 172800  # 2 days
        elif self.dataset_name == "enron":
            left = 86400  # 1 day
            mode = 172800  # 2 days
            right = 259200  # 3 days
        elif self.dataset_name == "bitcoin-otc":
            left = 86400  # 1 day
            mode = 172800  # 2 days
            right = 259200  # 3 days
        elif self.dataset_name == "wikipedia":
            left = 86400  # 1 day
            mode = 172800  # 2 days
            right = 259200  # 3 days
        else:
            raise NotImplementedError
        self.link_span_dist = partial(np.random.triangular, left, mode, right)

        # Do I actually need to check? It's fine to make it work for other datasets.. right?
        # assert "wsdm" in self.cdata.name

    def get_label_adj(self, i):
        start = i + 1
        end = start + self.timespan

        adj = tu.get_sp_adj(
            edges=self.cdata.edges,
            snapshot=end,
            weighted=True,
            time_window=self.timespan + 1,  # Time span is exclusive >, not >=
            temporal_granularity="continuous",
            with_edge_features=True,
        )
        idx = adj["idx"]
        time = adj["time"].squeeze()
        etype = adj["type"]

        min_time = time.min()
        max_time = time.max()

        # Sample links. If we select all links from the next 3 months, it might be too much.
        links_to_sample = int(self.max_links / 2)
        if time.size(0) < links_to_sample:
            pass
        else:
            # Sample with replacement
            # Assuming that it isn't a big issue if a link is selected multiple times.
            # This is in turn assuming that max_links << time.size(0)
            sample_idxs = torch.randint(0, time.size(0), (links_to_sample,))
            idx = idx[sample_idxs]
            time = time[sample_idxs]
            etype = etype[sample_idxs]

        # Split into two sets, links and no_links (negative samples)
        # Links are duplicated but the negative ones are given a different timestamp.
        links, no_links = idx, idx.detach().clone()
        link_type, no_link_type = etype, etype.detach().clone()
        link_time, no_link_time = time, time.detach().clone()

        # Randomize idx
        num_rand = int(idx.size(0) * self.args.rand_idx_rate)
        rand_links = torch.randint(0, idx.max() + 1, (num_rand, 2))
        # Add at the end of no_links to avoid randomizing index and etype on same entries
        no_links[idx.size(0) - num_rand : idx.size(0)] = rand_links

        # Randomize edge types
        num_rand = int(etype.size(0) * self.args.rand_etype_rate)
        rand_types = torch.randint(0, etype.max() + 1, (num_rand, 1))
        no_link_type[:num_rand] = rand_types

        # Suffle edge types
        rand = torch.randperm(no_link_type.size(0))
        no_link_type = no_link_type[rand]

        # Randomize link times
        num_rand = int(time.size(0) * self.args.rand_time_rate)
        rand_times = torch.randint(time.min(), time.max() + 1, (num_rand,))
        # Beginning or end doesn't matter, since they'll be suffled
        no_link_time[:num_rand] = rand_times

        # Suffle no link times
        # This is to create links that are likely to happen, but at the wrong time.
        # The time should also be at a reasonable time, but not for that link.
        rand = torch.randperm(no_link_time.size(0))
        no_link_time = no_link_time[rand]

        # Concat links and no links
        idx = torch.cat([links, no_links])
        vals = torch.cat(
            [torch.ones(link_time.size()[0]), torch.zeros(no_link_time.size()[0])]
        )
        time = torch.cat([link_time, no_link_time])
        etype = torch.cat([link_type, no_link_type])

        # # Add time spans around each link's time.
        # # This is to estimate whether the link will appear at this time or not.
        # time_span = self.link_span_dist(size=time.size()[0])
        # time_span_half = torch.from_numpy(time_span / 2).type(torch.int64)
        # time_start = time - time_span_half
        # time_end = time + time_span_half
        # time_start.clamp(min=min_time)
        # time_end.clamp(max=max_time)

        # # Convert times into time deltas (time spans)
        # start_delta = time_start - min_time
        # duration = time_end - time_start
        # time_query = torch.vstack((start_delta, duration)).t()

        # Prepare edge for decoder
        delta = time - min_time
        delta = delta.type(torch.FloatTensor)

        etype = etype.squeeze()
        edge_type = self.edge_type_to_one_hot(etype)
        time_feats = self.time_edge_features(time)

        label_adj = {
            "idx": idx,
            "vals": vals,
            "time": delta,
            "type": edge_type,
            "time_feats": time_feats,
        }

        return label_adj

    def to_one_hot(self, feat, max_val):
        num_rows = feat.shape[0]
        one_hot = torch.zeros([num_rows, max_val + 1])
        one_hot[range(num_rows), feat] = 1
        return one_hot

    def edge_type_to_one_hot(self, etype):
        if self.cdata.name == "wsdm-A":
            # Add edge type features
            # WSDM-A has 4 of them
            edge_type_feats = self.cdata.edge_type_features
            max_vals = edge_type_feats.max()

            edge_feats = edge_type_feats.iloc[etype]
            feat0 = self.to_one_hot(edge_feats[0].to_numpy(), max_vals[0])
            feat1 = self.to_one_hot(edge_feats[1].to_numpy(), max_vals[1])
            feat2 = self.to_one_hot(edge_feats[2].to_numpy(), max_vals[2])
            feat3 = self.to_one_hot(edge_feats[3].to_numpy(), max_vals[3])
            return torch.cat([feat0, feat1, feat2, feat3], dim=1)
        else:
            type_max_val = self.cdata.type_max_val
            return self.to_one_hot(etype, type_max_val)

    def time_edge_features(self, time):
        df = pd.DataFrame(time)
        df["dt"] = pd.to_datetime(df[0], unit="s")
        temp = df["dt"].to_frame()
        temp = temp.set_index("dt").index.to_series()
        df["dow"] = temp.dt.dayofweek.to_frame().set_index(df.index)
        df["hod"] = temp.dt.hour.to_frame().set_index(df.index)

        dow = self.to_one_hot(df["dow"].to_numpy(), 6)
        hod = self.to_one_hot(df["hod"].to_numpy(), 23)
        return torch.cat([dow, hod], dim=1)
