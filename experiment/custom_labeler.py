import pandas as pd
import torch
from tlp_labeler import Tlp_Labeler


class Custom_Labeler:
    def __init__(self, args, continuous_dataset, settype="initial"):

        self.data_folder = "data"
        self.get_data = None  # A function that takes an idx and returns a label_adj
        self.datasetname = continuous_dataset.name
        self.settype = settype
        self.tlp_labeler = Tlp_Labeler(args, continuous_dataset)

        self._load_wsdm_data(continuous_dataset, self.datasetname, settype=settype)

    def _load_wsdm_data(self, dataset, datasetname, settype="initial"):
        settypefilename = settype
        assert datasetname in ["wsdm-A", "wsdm-B", "enron"]
        if datasetname == "wsdm-A":
            filename = "input_A_{}.csv.gz".format(settypefilename)
            filepath = "{}/wsdm-A/{}".format(self.data_folder, filename)
        else:
            filename = "input_B_{}.csv.gz".format(settypefilename)
            filepath = "{}/wsdm-B/{}".format(self.data_folder, filename)

        df = pd.read_csv(filepath, compression="gzip", header=None)

        if settype == "intermediate" or settype == "test":
            # Add dummy labels
            assert 5 not in df.columns
            df[5] = 0

        df = df.rename(
            columns={
                0: "src",
                1: "target",
                2: "type",
                3: "start_time",
                4: "end_time",
                5: "label",
            }
        )

        # print("Remove below line")
        # df = df[df["type"] <= 5]
        # print(df)
        # print(df["type"])
        # print(df["type"].max())

        translator = dataset.node_id_translator
        if translator != None:
            df["src"] = df["src"].map(translator["old2new"])
            df["target"] = df["target"].map(translator["old2new"])

        links = df[["src", "target"]].to_numpy()
        etype = df["type"].to_numpy()
        df["time"] = df["end_time"] - (df["end_time"] - df["start_time"]) / 2
        time = df["time"].to_numpy()
        labels = df["label"].to_numpy()

        idx = torch.Tensor(links).type(torch.int64)
        vals = torch.Tensor(labels).type(torch.uint8)
        etype = torch.Tensor(etype).type(torch.int64)
        time = torch.Tensor(time).float()

        min_time = time.min()
        delta = time - min_time

        edge_type = self.tlp_labeler.edge_type_to_one_hot(etype)
        time_feats = self.tlp_labeler.time_edge_features(time)

        label_adj = {
            "idx": idx,
            "vals": vals,
            "time": delta,
            "type": edge_type,
            "time_feats": time_feats,
        }

        def get_data(idx):
            return label_adj

        self.get_data = get_data

    def get_label_adj(self, idx):
        return self.get_data(idx)
