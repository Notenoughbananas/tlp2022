import utils as u


class Dataset:
    def __init__(
        self,
        edgesidx,
        vals,
        node_features=None,
        edge_features=None,
        node_id_translator=None,
        name=None,
        edge_type_features=None,
    ):
        self.name = name
        # Continuous models may use the edge features
        # If discrete models use the vals (edge weights).
        self.edges = {"idx": edgesidx, "vals": vals}
        self.node_features = node_features
        self.edge_features = edge_features
        self.edge_type_features = edge_type_features

        # Used to translate from node indices to reindexed node indices
        self.node_id_translator = node_id_translator

        self.cols = u.Namespace(
            {"source": 0, "target": 1, "time": 2, "snapshot": 3, "type": 4}
        )
        self.num_nodes = int(
            edgesidx[:, [self.cols.source, self.cols.target]].max() + 1
        )
        # Check if node ids are contiguous and zero indexed
        # assert(len(edgesidx[:, [self.cols.source, self.cols.target]].unique()) == self.num_nodes)
        self.max_time = edgesidx[:, self.cols.snapshot].max()
        self.min_time = edgesidx[:, self.cols.snapshot].min()
        self.num_snapshots = self.max_time - self.min_time

        # Used only for outputting dataset information
        self.num_edges = edgesidx.shape[0]

        # Different for discrete and continuous since the discrete have reciprocal links added
        self.num_unique_edges = len(
            set(
                tuple(edge)
                for edge in edgesidx[:, [self.cols.source, self.cols.target]]
                .squeeze()
                .tolist()
            )
        )

        self.mean_unique_edges_per_snapshot = self.num_edges / float(self.num_snapshots)
        self.mean_snapshot_density = self.mean_unique_edges_per_snapshot / (
            self.num_nodes * (self.num_nodes - 1)
        )
        self.density = self.num_unique_edges / (self.num_nodes * (self.num_nodes - 1))

        self.max_time_real = edgesidx[:, self.cols.time].max()
        self.min_time_real = edgesidx[:, self.cols.time].min()
        self.duration = self.max_time_real - self.min_time_real

        if edgesidx.size(1) >= 5:
            # Has type
            self.type_max_val = edgesidx[:, self.cols.type].max()
