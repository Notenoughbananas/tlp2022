import torch
import utils as u
import numpy as np
import pandas as pd

from datareaders.dataset import Dataset


def normalize_time(time_vector):
    return time_vector - time_vector.min()


# Consider renaming to snapshotify for fun.
def aggregate_by_time(time_vector, time_win_aggr):
    return time_vector // time_win_aggr


# Not used? Remove?
def edges_to_dataset(edges, ecols):
    idx = edges[:, [ecols.source, ecols.target, ecols.time, ecols.snapshot]]

    vals = edges[:, ecols.weight]
    return Dataset(idx, vals)


def tgat_preprocess(data_name):
    # Wikipedia and Reddit - Preprocessing code kindly borrowed from TGAT.
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(",")
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame(
        {"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list}
    ), np.array(feat_l)


def tgat_reindex(df, bipartite=True):
    # Improved TGAT reindexing borrowed from TGN
    new_df = df.copy()
    if bipartite:
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


# If strictly_evolving=True, return a strictly evolving version of the discrete dataset.
def continuous2discrete(continuous: Dataset, strictly_evolving=False) -> Dataset:

    # Remove time col
    discrete_edgeidx = continuous.edges["idx"].detach().clone()
    discrete_edgeidx = discrete_edgeidx[
        :, [continuous.cols.source, continuous.cols.target, continuous.cols.snapshot]
    ]
    # add the reversed link to make the graph undirected. This is also required for the
    #  uniqueness check to make sense. Because it would count a,b as a different edge to b,a.
    #  for an edge between node a and b.
    cols = u.Namespace({"source": 0, "target": 1, "snapshot": 2})
    discrete_edgeidx = torch.cat(
        [
            discrete_edgeidx,
            discrete_edgeidx[:, [cols.target, cols.source, cols.snapshot]],
        ]
    )

    if strictly_evolving:
        discrete_edgeidx = to_strictly_evolving(discrete_edgeidx, cols)

    # Unique is memory heavy for large datasets, therefore we compute it snapshot by snapshot

    disc_edgeidx_l = []
    disc_vals_l = []
    for ss in range(0, discrete_edgeidx[:, 2].max() + 1):
        # Returns unique (source, target, snapshot) triplets.
        # We therefore get one unique edge per snapshot
        # The count is therefore the number of times an edge occur in each snapshot
        snapshot = discrete_edgeidx[discrete_edgeidx[:, 2] == ss]
        ss_discrete_edgeidx, ss_discrete_vals = snapshot.unique(
            sorted=False, return_counts=True, dim=0
        )
        disc_edgeidx_l.append(ss_discrete_edgeidx)
        disc_vals_l.append(ss_discrete_vals)

    discrete_edgeidx = torch.cat(disc_edgeidx_l, dim=0)
    discrete_vals = torch.cat(disc_vals_l, dim=0)

    # Duplicate the snapshot index, thus the snapshot index is now in the time AND snapshot col
    discrete_edgeidx = torch.cat(
        [discrete_edgeidx, discrete_edgeidx[:, 2].view(-1, 1)], dim=1
    )

    return Dataset(
        discrete_edgeidx,
        discrete_vals,
        node_features=continuous.node_features,
        node_id_translator=continuous.node_id_translator,
        name=continuous.name,
    )


def to_strictly_evolving(edgesidx, cols):
    max_time = edgesidx[:, cols.snapshot].max()
    min_time = edgesidx[:, cols.snapshot].min()

    subset = edgesidx[:, cols.snapshot] == min_time
    prev_snapshot = edgesidx[subset]

    evolving_snapshots = [prev_snapshot]
    # Looks at previous snapshot and carry over links from it.
    for snapshot in range(min_time + 1, max_time + 1):
        subset = edgesidx[:, cols.snapshot] == snapshot
        current_and_prev_ss = torch.cat([prev_snapshot, edgesidx[subset]], dim=0)
        new_edgesidx = current_and_prev_ss[:, [cols.source, cols.target]].unique(
            sorted=False, dim=0
        )
        time = torch.ones(new_edgesidx.size(0), 1, dtype=torch.long) * snapshot
        new_snapshot = torch.cat([new_edgesidx, time], dim=1)
        evolving_snapshots.append(new_snapshot)
        prev_snapshot = new_snapshot

    return torch.cat(evolving_snapshots)


def dataset2snapshot_set(dataset, snapshotidx):
    edgeidx = dataset.edges["idx"]
    cols = dataset.cols
    snapshot_mask = edgeidx[:, cols.snapshot] == snapshotidx
    snapshot = edgeidx[snapshot_mask][:, [cols.source, cols.target]]
    return set(tuple(edge) for edge in snapshot.cpu().numpy())


def persistance_rate(dataset):
    rates = []

    prev_ss_set = dataset2snapshot_set(dataset, dataset.min_time)
    for snapshotidx in range(dataset.min_time + 1, dataset.max_time):
        curr_ss_set = dataset2snapshot_set(dataset, snapshotidx)
        persisting_edges = prev_ss_set & curr_ss_set

        num_prev = float(len(prev_ss_set))
        num_persisting = float(len(persisting_edges))
        rate = num_persisting / num_prev
        rates.append(rate)
        prev_ss_set = curr_ss_set

    rates = np.array(rates)
    return np.mean(rates), np.std(rates)


def recurrence_rate(dataset):
    rates = []
    edge_memory = dataset2snapshot_set(
        dataset, dataset.min_time
    )  # Remember edges that existed once
    prev_ss = dataset2snapshot_set(dataset, dataset.min_time)

    for snapshotidx in range(dataset.min_time + 1, dataset.max_time):
        curr_ss = dataset2snapshot_set(dataset, snapshotidx)
        reoccuring_edges = edge_memory & (curr_ss - prev_ss)
        assert (reoccuring_edges | edge_memory) == edge_memory
        assert (reoccuring_edges | curr_ss) == curr_ss
        rates.append(len(reoccuring_edges) / float(len(edge_memory)))

        edge_memory = edge_memory | curr_ss  # Update memory
        prev_ss = curr_ss
    rates = np.array(rates)
    return np.mean(rates), np.std(rates)


# Makes the number of occurrences of a link in a snapshot the weight of that link
# Coalesce sums the values of identical indexes together
# Thus make sure that time col is not included in index, if not this has no effect.
def weight_by_occurrece(index, occurences, num_nodes, max_time):
    return torch.sparse.LongTensor(
        index, occurences, torch.Size([num_nodes, num_nodes, max_time + 1])
    ).coalesce()


def make_contiguous_node_ids(edges, ecols, return_translator=False):
    new_edges = edges[:, [ecols.source, ecols.target]]
    node_idxs, new_edges = new_edges.unique(return_inverse=True)
    edges[:, [ecols.source, ecols.target]] = new_edges
    if not return_translator:
        return edges
    else:
        old2new = {}
        new2old = {}
        for i, key in enumerate(node_idxs.numpy()):
            old2new[key] = i
            new2old[i] = key
        # old2new = {key: i for i, key in enumerate(node_idxs.numpy())}
        translator = {"old2new": old2new, "new2old": new2old}
        return translator, edges


def load_edges_as_tensor(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()
    edges = [[float(r) for r in row.split(",")] for row in lines]
    edges = torch.tensor(edges, dtype=torch.long)
    return edges


def weekday_filter(args, df):
    # Assumes the df already has a datetime column named "dt"

    def add_day_of_week_column(df):
        temp = df["dt"].to_frame()
        index = temp.index
        temp = temp.set_index("dt").index.to_series()

        # Monday = 0, Sunday = 6
        df["dow"] = temp.dt.dayofweek.to_frame().set_index(index)
        return df

    # Split weekend and weekdays
    if not hasattr(args, "data_filter"):
        # No split
        pass
    elif args.weekday_filter == "weekdays":
        # Filter out weekends
        df = add_day_of_week_column(df)
        df = df[(df["dow"] < 5)]  # Saturday = 5
        del df["dow"]

    elif args.weekday_filter == "weekends":
        # Filter out weekdays
        df = add_day_of_week_column(df)
        df = df[(df["dow"] >= 5)]  # Saturday = 5
        del df["dow"]

    else:
        print("No weekend, weekday split")
        # No split
        pass

    return df
