import torch
import os, sys
import pandas as pd
import numpy as np
import utils as u
import tarfile
from datetime import datetime
import datareaders.data_utils as du
from datareaders.dataset import Dataset


class Datareader:
    def __init__(self, args):
        self.args = args
        # return_continuous = args.temporal_granularity == 'continuous' #Returns both by default

        # Convert snapshot size from days to seconds
        dataset_name = args.data
        if dataset_name != "autonomous-systems":
            self.args.snapshot_size = (
                60 * 60 * 24 * self.args.snapshot_size
            )  # days to seconds
        self.args.snapshot_size = int(self.args.snapshot_size)
        self.strictly_evolving = (
            hasattr(args, "strictly_evolving") and args.strictly_evolving == True
        )

        if dataset_name == "enron":
            dataset = self.load_enron(dataset_name)
        elif dataset_name in ["bitcoin-alpha", "bitcoin-otc"]:
            dataset = self.load_bitcoin(dataset_name)
        elif dataset_name == "autonomous-systems":
            steps_accounted = self.args.steps_accounted
            dataset = self.load_autosys(dataset_name, steps_accounted)
        elif dataset_name == "uc":
            edges_file = "opsahl-ucsocial/out.opsahl-ucsocial"
            dataset = self.load_uc(dataset_name, edges_file)
        elif dataset_name == "wikipedia":
            dataset = self.load_tgat_data(dataset_name)
        elif dataset_name == "reddit":
            dataset = self.load_tgat_data(dataset_name)
        elif dataset_name == "wsdm-A":
            # self.max_links = self.args.max_continuous_links
            dataset = self.load_wsdm_A(dataset_name)
        elif dataset_name == "wsdm-B":
            dataset = self.load_wsdm_B(dataset_name)
        else:
            raise ValueError("Dataset {} not found".format(self.args.data))

        # self.print_all_dataset_info_and_exit()
        self.dataset = dataset

    # Returns a dataframe with columns: {source, target, t, snapshot}
    # First event is always at t=0
    def load_enron(self, dataset_name) -> Dataset:
        snapshot_size = self.args.snapshot_size
        filepath = self.args.data_filepath
        # The "whatisthis" column is just filled with 1 so it has no impact and can safely be removed.
        df = pd.read_csv(
            filepath, header=None, names=["source", "target", "ones", "t"], sep=" "
        )
        df = df.drop("ones", axis=1)

        # Make t relative. I.e. set the first event to be at t=0
        df["t"] = df["t"] - df["t"].min()
        # Snapshot by day (divide by seconds in a day)
        df["snapshot"] = (df["t"] / snapshot_size).apply(int)
        assert df["t"].is_monotonic_increasing

        # First node should be node 0, not node 1.
        df[["source", "target"]] = df[["source", "target"]] - 1
        df["type"] = 5  # Just picked at random
        edgesidx = torch.from_numpy(df.to_numpy())

        # edgesidx[:, 4] = torch.from_numpy(np.ones(edgesidx.size(0))*edge_type)

        continuous = Dataset(edgesidx, torch.ones(edgesidx.size(0)), name=dataset_name)
        return continuous, du.continuous2discrete(
            continuous,
            strictly_evolving=self.strictly_evolving,
        )

    def load_bitcoin(self, dataset_name) -> Dataset:
        snapshot_size = self.args.snapshot_size
        filepath = self.args.data_filepath
        ecols = u.Namespace(
            {"source": 0, "target": 1, "weight": 2, "time": 3, "snapshot": 4}
        )

        def cluster_negs_and_positives(ratings):
            pos_indices = ratings > 0
            neg_indices = ratings <= 0
            ratings[pos_indices] = 1
            ratings[neg_indices] = -1
            return ratings

        # build edge data structure
        edges = du.load_edges_as_tensor(filepath)

        edges = du.make_contiguous_node_ids(edges, ecols)

        edges[:, ecols.time] = du.normalize_time(edges[:, ecols.time])
        snapshots = du.aggregate_by_time(edges[:, ecols.time], snapshot_size)
        edges = torch.cat([edges, snapshots.view(-1, 1)], dim=1)

        edges[:, ecols.weight] = cluster_negs_and_positives(edges[:, ecols.weight])

        # Sort edges by time. Ensure that they are in the correct temporal order.
        # Essential for bitcoin alpha on TGN.
        edges = edges[edges[:, ecols.time].sort()[1]]

        continuous = Dataset(
            edges[:, [ecols.source, ecols.target, ecols.time, ecols.snapshot]],
            edges[:, ecols.weight],
            name=dataset_name,
        )

        return continuous, du.continuous2discrete(
            continuous, strictly_evolving=self.strictly_evolving
        )

    def load_autosys(self, dataset_name, steps_accounted: int) -> Dataset:
        snapshot_size = self.args.snapshot_size
        tar_file = self.args.data_filepath

        def times_from_names(files):
            files2times = {}
            times2files = {}

            base = datetime.strptime("19800101", "%Y%m%d")
            for file in files:
                delta = (datetime.strptime(file[2:-4], "%Y%m%d") - base).days

                files2times[file] = delta
                times2files[delta] = file

            cont_files2times = {}

            sorted_times = sorted(files2times.values())
            new_t = 0

            for t in sorted_times:

                file = times2files[t]

                cont_files2times[file] = new_t

                new_t += 1
            return cont_files2times

        tar_archive = tarfile.open(tar_file, "r:gz")
        files = tar_archive.getnames()

        cont_files2times = times_from_names(files)

        edges = []
        cols = u.Namespace({"source": 0, "target": 1, "time": 2, "snapshot": 3})
        for file in files:
            data = u.load_data_from_tar(
                file,
                tar_archive,
                starting_line=4,
                sep="\t",
                type_fn=int,
                tensor_const=torch.LongTensor,
            )

            # Is this really the correct way to turn a dict into a tensor?
            time_col = (
                torch.zeros(data.size(0), 1, dtype=torch.long) + cont_files2times[file]
            )

            # This double time col thing is not pretty, but it seems to work.
            data = torch.cat([data, time_col, time_col], dim=1)
            data = torch.cat(
                [data, data[:, [cols.target, cols.source, cols.time, cols.snapshot]]]
            )

            edges.append(data)

        edges.reverse()
        edges = torch.cat(edges)

        edges = du.make_contiguous_node_ids(edges, cols)

        # use only first X time steps
        # NOTE: This filters away some nodes and the node ids are no longer contiguous.
        # Max node counts and such are based on this and the decoder will now train on nodes which are not in the selected snapshots.
        # Chose to leave du.make_contiguous_node_ids above this step to remain comparable to EvolveGCN results. However, consider moving it below in the future to save time.

        indices = edges[:, cols.snapshot] < steps_accounted
        edges = edges[indices, :]

        # Snapshot col already added and time already normalized by previous code
        edges[:, cols.snapshot] = du.aggregate_by_time(
            edges[:, cols.time], snapshot_size
        )

        continuous = Dataset(edges, torch.ones(edges.size(0)), name=dataset_name)

        return continuous, du.continuous2discrete(
            continuous, strictly_evolving=self.strictly_evolving
        )

    def load_uc(self, dataset_name, edges_file: str) -> Dataset:
        snapshot_size = self.args.snapshot_size
        tar_file = self.args.data_filepath

        with tarfile.open(tar_file, "r:bz2") as tar_archive:
            data = u.load_data_from_tar(
                edges_file, tar_archive, starting_line=2, sep=" "
            )
        edges = data.long()

        cols = u.Namespace(
            {"source": 0, "target": 1, "weight": 2, "time": 3, "snapshot": 4}
        )
        # first id should be 0 (they are already contiguous)
        edges[:, [cols.source, cols.target]] -= 1

        edges[:, cols.time] = du.normalize_time(edges[:, cols.time])
        snapshots = du.aggregate_by_time(edges[:, cols.time], snapshot_size)
        edges = torch.cat([edges, snapshots.view(-1, 1)], dim=1)

        continuous = Dataset(
            edges[:, [cols.source, cols.target, cols.time, cols.snapshot]],
            edges[:, cols.weight],
            name=dataset_name,
        )

        return continuous, du.continuous2discrete(
            continuous, strictly_evolving=self.strictly_evolving
        )

    def load_tgat_data(self, dataset_name) -> Dataset:
        snapshot_size = self.args.snapshot_size
        df, feat = du.tgat_preprocess(self.args.data_filepath)
        df = du.tgat_reindex(df, bipartite=True)
        df["ss"] = (df["ts"] / snapshot_size).apply(int)
        df = df[["u", "i", "ts", "ss"]]
        # First node should be node 0, not node 1.
        df[["u", "i"]] = df[["u", "i"]] - 1

        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        edge_features = np.vstack([empty, feat])

        edgesidx = torch.from_numpy(df.to_numpy()).long()
        continuous = Dataset(
            edgesidx, torch.zeros(edgesidx.size(0)), edge_features, name=dataset_name
        )  # , dtype=torch.long))

        return continuous, du.continuous2discrete(
            continuous, strictly_evolving=self.strictly_evolving
        )

    def load_wsdm_A(self, dataset_name) -> Dataset:
        snapshot_size = self.args.snapshot_size
        folder = self.args.data_filepath

        edgesfile = "{}/edges_train_A.csv.gz".format(folder)
        nodefeatsfile = "{}/node_features.csv.gz".format(folder)
        edgefeatsfile = "{}/edge_type_features.csv.gz".format(folder)

        df = pd.read_csv(
            edgesfile,
            compression="gzip",
            names=["source", "target", "type", "t"],
            header=None,
        )
        node_feats = pd.read_csv(nodefeatsfile, compression="gzip", header=None)
        edge_type_feats = pd.read_csv(edgefeatsfile, compression="gzip", header=None)

        # edge_type = df["type"]
        # #del df["type"]

        # # # Edge features
        # # # Simply "unpack" edge features so we don't look at edge type anymore
        # edge_features = edge_type_feats.loc[edge_type, :].to_numpy()
        # # 2. One hot encode edge features
        # max_val = edge_features.max()
        # num_rows = edge_features[:, 0].shape[0]
        # one_hot = np.zeros([num_rows, max_val + 1])

        # # Add each feature to the same vector. Thus there will be 4 ones in this vector
        # one_hot[range(num_rows), edge_features[:, 0]] = 1
        # one_hot[range(num_rows), edge_features[:, 1]] = 1
        # one_hot[range(num_rows), edge_features[:, 2]] = 1
        # one_hot[range(num_rows), edge_features[:, 3]] = 1
        # df["type"] = one_hot

        # Correct time
        # It seems like the weekend gap is off by about 24 hours in this dataset.
        # This corrects this, for a better weekday/weekend split
        df["t"] = df["t"] - 86400

        # Start of the next Monday at midnight
        df["dt"] = pd.to_datetime(df["t"], unit="s")
        df = df[~(df["dt"] < "2014-10-20")]

        # Filter data. Split between weekdays and weekends
        df = du.weekday_filter(self.args, df)
        del df["dt"]

        # Make t relative. I.e. set the first event to be at t=0
        df["t"] = df["t"] - df["t"].min()
        # Snapshot by day (divide by seconds in a day)
        df["snapshot"] = (df["t"] / snapshot_size).apply(int)
        assert df["t"].is_monotonic_increasing

        # Reorder columns
        df = df[["source", "target", "t", "snapshot", "type"]]

        edges = torch.from_numpy(df.to_numpy())
        cols = u.Namespace(
            {"source": 0, "target": 1, "time": 2, "snapshot": 3, "type": 4}
        )
        node_id_translator, edges = du.make_contiguous_node_ids(
            edges, cols, return_translator=True
        )

        # Node features
        node_feats[0] = node_feats[0].map(node_id_translator["old2new"])
        node_feats = node_feats.set_index(0).to_numpy()

        continuous = Dataset(
            edges,
            torch.ones(edges.size(0)),
            node_features=node_feats,
            edge_type_features=edge_type_feats,
            node_id_translator=node_id_translator,
            name=dataset_name,
        )
        discrete = du.continuous2discrete(
            continuous, strictly_evolving=self.strictly_evolving
        )

        # # Continuous dataset Preprocessing
        # # 1. The continuous dataset has simply too many edges. We solve this my randomly sampling edges

        # # Sample with replacement
        # # Assuming that it isn't a big issue if a link is selected multiple times.
        # # This is in turn assuming that max_links << time.size(0)
        # if edges.size(0) > self.max_links:
        #     sample_idxs = torch.randint(0, edges.size(0), (self.max_links,))
        #     edges = edges[sample_idxs]
        #     edge_features = edge_features[sample_idxs]

        # continuous = Dataset(
        #    edges,
        #    torch.ones(edges.size(0)),
        #    node_features=node_feats,
        #    node_id_translator=node_id_translator,
        #    name=dataset_name,
        # )

        return continuous, discrete

    def load_wsdm_B(self, dataset_name) -> Dataset:
        snapshot_size = self.args.snapshot_size
        folder = self.args.data_filepath

        edgesfile = "{}/edges_train_B.csv.gz".format(folder)
        df = pd.read_csv(
            edgesfile,
            compression="gzip",
            names=["source", "target", "type", "t", "feats"],
            header=None,
        )

        # BERT like features
        # edges_with_features = df.dropna()
        del df["feats"]

        # Start on Monday at Midnight. Filter out any date before this time
        df["dt"] = pd.to_datetime(df["t"], unit="s").to_frame()
        df = df[~(df["dt"] < "2015-01-05")]

        # Filter data. Split between weekdays and weekends
        df = du.weekday_filter(self.args, df)
        del df["dt"]

        ## Make temporary series with datetime index to get dayofweek and hourofday
        # temp = df["dt"].to_frame()
        # index = temp.index
        # temp = temp.set_index("dt").index.to_series()
        # df['dow'] = temp.dt.dayofweek.to_frame().set_index(index)
        # df['hod'] = temp.dt.hour.to_frame().set_index(index)

        # Make t relative. I.e. set the first event to be at t=0
        df["t"] = df["t"] - df["t"].min()
        # Snapshot by day (divide by seconds in a day)
        df["snapshot"] = (df["t"] / snapshot_size).apply(int)

        # # One hot encode edge type
        # edge_type = df["type"].to_numpy()
        # max_val = edge_type.max()
        # num_rows = edge_type.shape[0]
        # edge_feat_one_hot = np.zeros([num_rows, max_val + 1])
        # edge_feat_one_hot[range(num_rows), edge_type] = 1
        # df["type"] = edge_feat_one_hot
        # del df["type"]

        ## There are only 14 edge types, so we make the vector bigger based on the layer size.
        ## The feature vectors are enlarged by adding zeros.
        # layer_feat_size = self.args.gcn_parameters["layer_1_feats"]
        # if layer_feat_size > max_val + 1:
        #    zero_feats = np.zeros(
        #        [
        #            edge_feat_one_hot.shape[0],
        #            layer_feat_size - edge_feat_one_hot.shape[1],
        #        ]
        #    )
        #    edge_feat_one_hot = np.concatenate([edge_feat_one_hot, zero_feats], axis=1)

        # cont_edge_features = edges_with_features["feats"]
        # del edges_with_features["feats"]

        assert df["t"].is_monotonic_increasing

        # Reorder columns
        df = df[["source", "target", "t", "snapshot", "type"]]
        edges = torch.from_numpy(df.to_numpy())

        # BERT like features
        ## Currently ignoring edge features for now.
        ## This will be useful if we get to train combined
        # cont_edges = torch.from_numpy(edges_with_features.to_numpy())
        # cont_feats = cont_edge_features.to_numpy()

        # continuous_only_edges_w_feats = Dataset(
        #     cont_edges,
        #     torch.ones(cont_edges.size(0)),
        #     edge_features=cont_feats,
        #     name=dataset_name,
        # )

        # # Sample with replacement
        # # Assuming that it isn't a big issue if a link is selected multiple times.
        # # This is in turn assuming that max_links << time.size(0)
        # if edges.size(0) > self.max_links:
        #     sample_idxs = torch.randint(0, edges.size(0), (self.max_links,))
        #     edges = edges[sample_idxs]
        #     edge_feat_one_hot = edge_feat_one_hot[sample_idxs]

        continuous = Dataset(
            edges,
            torch.ones(edges.size(0)),
            # edge_features=edge_feat_one_hot,
            name=dataset_name,
        )
        discrete = du.continuous2discrete(
            continuous, strictly_evolving=self.strictly_evolving
        )
        return continuous, discrete

    def print_all_dataset_info_and_exit(self):
        dataset_names = [
            "enron",
            "bitcoin-otc",
            "bitcoin-alpha",
            "autonomous-systems",
            "uc",
            "wikipedia",
            "reddit",
            "wsdm-A",
            "wsdm-B",
        ]
        print(
            "|Name|num nodes|ss density|E|uniq E | density | num ss| cont E | cont uniq E| cont density |duration|cont density? | persistance | recurrance |"
        )

        for dataset_name in dataset_names:
            from run_exp import read_data_master

            self.args = read_data_master(self.args, dataset_name=dataset_name)
            if dataset_name != "autonomous-systems":
                self.args.snapshot_size = (
                    60 * 60 * 24 * self.args.snapshot_size
                )  # days to seconds
            self.args.snapshot_size = int(self.args.snapshot_size)
            if dataset_name == "enron":
                cdataset, dataset = self.load_enron()
            elif dataset_name in ["bitcoin-otc", "bitcoin-alpha"]:
                cdataset, dataset = self.load_bitcoin()
            elif dataset_name == "autonomous-systems":
                steps_accounted = self.args.steps_accounted
                cdataset, dataset = self.load_autosys(steps_accounted)
            elif dataset_name == "uc":
                edges_file = "opsahl-ucsocial/out.opsahl-ucsocial"
                cdataset, dataset = self.load_uc(edges_file)
            elif dataset_name == "wikipedia":
                cdataset, dataset = self.load_tgat_data()
            elif dataset_name == "reddit":
                cdataset, dataset = self.load_tgat_data()
            elif dataset_name == "wsdm-A":
                cdataset, dataset = self.load_wsdm_A("wsdm-A")
            elif dataset_name == "wsdm-B":
                cdataset, dataset = self.load_wsdm_B("wsdm-A")
            else:
                raise ValueError("Dataset {} not found".format(self.args.data))

            if dataset_name != "autonomous-systems":
                duration_in_days = int(cdataset.duration / (60.0 * 60 * 24))
            else:
                duration_in_days = cdataset.duration

            persistance_rate, persistance_std = du.persistance_rate(dataset)
            recurrence_rate, recurrence_std = du.recurrence_rate(dataset)

            print(
                "|{}|{}|{}|{}|{}|{:.4f}|{}|{}|{}|{:.4f}|{}|{:d}|{:.3f}({:.3f})|{:.3f}({:.3f})".format(
                    dataset_name,
                    dataset.num_nodes,
                    dataset.mean_snapshot_density,
                    int(
                        dataset.num_edges / 2
                    ),  # Divided by two because these are directed edges that have reciprocial edges added.
                    int(
                        dataset.num_unique_edges / 2
                    ),  # Divided by two because these are directed edges that have reciprocial edges added.
                    dataset.density,
                    dataset.num_snapshots,
                    cdataset.num_edges,
                    cdataset.num_unique_edges,
                    cdataset.density,
                    duration_in_days,
                    round(cdataset.num_edges / cdataset.num_nodes),
                    persistance_rate,
                    persistance_std,
                    recurrence_rate,
                    recurrence_std,
                )
            )
        sys.exit(0)
