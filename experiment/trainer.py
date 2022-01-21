import os
import functools
import utils as u
import logger
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import torch
import math
import os.path
from torch.nn import BCEWithLogitsLoss
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import NotFittedError

class Trainer:
    def __init__(
        self,
        args,
        splitter,
        final_splitter,
        comp_loss,
        dataset,
        disc_encoder=None,
        cont_encoder=None,
        classifier=None,
        combined_splitter=None,
        downstream_splitter=None,
        train_encoder=True,
    ):
        self.final_epoch_id = 9999
        self.args = args
        self.dataset = dataset
        self.splitter = splitter
        self.combined_splitter = combined_splitter
        self.downstream_splitter = downstream_splitter
        self.final_splitter = final_splitter
        self.tasker = splitter.tasker
        if combined_splitter != None:
            self.disc_tasker = combined_splitter.disc_tasker
        else:
            self.disc_tasker = None
        self.disc_encoder = disc_encoder
        self.cont_encoder = cont_encoder
        self.classifier = classifier
        self.downstream_classifiers = {
            "logistic": LogisticRegression(),
            "decision": DecisionTreeClassifier(),
            "xgb": XGBClassifier(),
        }
        self.comp_loss = comp_loss
        self.interpolation_loss = torch.nn.BCELoss()  # For continuous DGNN

        self.num_nodes = self.tasker.data.num_nodes
        self.num_classes = self.tasker.num_classes

        self.train_encoder = (
            train_encoder  # Indicate which kind of training this class is for
        )
        self.init_optimizers(args)

        self.downstream = False  # Used to keep track of whether we're downstream or not
        self.frozen = False  # Used to keep track of whether the encoder is frozen
        self.set_save_predictions = (
            False  # Keep track of whether to save predictions or not
        )
        self.has_time_query = (
            hasattr(self.args, "has_time_query") and self.args.has_time_query == True
        )

        self.use_tgn_memory = (
            args.model == "tgn" and self.args.gcn_parameters["use_memory"] == True
        )
        self.tgn_train_memory_backup = None
        self.tgn_val_memory_backup = None
        self.embedding_cache = {"TRAIN": {}, "VALID": {}, "TEST": {}}

        # Runs using a logfile are usually more serious and we'll like to keep those checkpoints.
        # Runs not using a logfile store the checkpoints in the working dir where they may
        # later be overwritten
        if args.use_logfile:
            if args.temporal_granularity == "continuous":
                self.checkpoint_filename_prefix = (
                    "checkpoints/{}-{}-learning_rate{}".format(
                        args.data, args.model, args.learning_rate
                    )
                )
            else:
                self.checkpoint_filename_prefix = "checkpoints/{}-{}-{}-".format(
                    args.data, args.model, args.grid
                )
            prediction_folder = "predictions/{}-{}/".format(args.data, args.model)
            self.prediction_filename_prefix = "{}{}-".format(
                prediction_folder, u.get_gridcell(args)
            )
            os.makedirs(prediction_folder, exist_ok=True)
        else:
            self.checkpoint_filename_prefix = "checkpoints/"
            # self.checkpoint_filename_prefix = 'wikipedia-tgat-'
            self.prediction_filename_prefix = "predictions/"

        # if self.tasker.is_static:
        #    adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
        #    self.hist_adj = [adj_matrix]
        #    self.hist_ndFeats = [self.tasker.nodes_feats.float()]

    def report_trainable_parameters(self):
        # The "requires_grad if test" seem to make no difference, but it's kept in just in case and it doesn't hurt.
        count_trainable_parameters = lambda model: sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
        )

        num_disc_encoder_params = 0
        num_cont_encoder_params = 0
        num_classifier_params = 0
        if self.disc_encoder != None:
            num_disc_encoder_params = count_trainable_parameters(self.disc_encoder)
        if self.cont_encoder != None:
            num_cont_encoder_params = count_trainable_parameters(self.cont_encoder)
        if self.classifier != None:
            num_classifier_params = count_trainable_parameters(self.classifier)

        num_encoder_params = num_disc_encoder_params + num_cont_encoder_params
        num_total_params = num_encoder_params + num_classifier_params

        if self.disc_tasker != None:
            disc_node_feats = self.disc_tasker.feats_per_node
        else:
            disc_node_feats = self.tasker.feats_per_node

        cont_node_feats = self.args.gcn_parameters["layer_2_feats"]
        if self.args.temporal_granularity in ["static", "discrete"]:
            node_feats = disc_node_feats
        else:
            node_feats = self.args.gcn_parameters["layer_2_feats"]
        num_parameter_str = "Number of parameters (i) Encoder(s): {}, (ii) Classifier: {}, Total: {}, features per node {}".format(
            num_encoder_params, num_classifier_params, num_total_params, node_feats
        )
        print(num_parameter_str)
        self.logger.logger.info(num_parameter_str)

    def init_optimizers(self, args):
        if self.disc_encoder != None and self.args.model != "random":
            params = self.disc_encoder.parameters()
            self.opt_encoder = torch.optim.Adam(params, lr=args.learning_rate)
            # , weight_decay=args.weight_decay)
            self.opt_encoder.zero_grad()

        if self.cont_encoder != None:
            params = self.cont_encoder.parameters()
            self.opt_cont_encoder = torch.optim.Adam(params, lr=args.learning_rate)
            # , weight_decay=args.weight_decay)
            self.opt_cont_encoder.zero_grad()

        if self.classifier != None:
            params = self.classifier.parameters()
            if self.train_encoder:
                self.opt_decoder = torch.optim.Adam(
                    params, lr=args.learning_rate
                )  # , weight_decay=args.weight_decay)
            else:
                # If we train only the decoder we want a specific learning rate for it and regularization on it.
                self.opt_decoder = torch.optim.Adam(
                    params,
                    lr=args.decoder_learning_rate,
                    weight_decay=args.decoder_weight_decay,
                )
            self.opt_decoder.zero_grad()

    # Checks all checkpoints, if one is missing for one of the encoders, then no is returned.
    def checkpoint_exists(self):
        def encoder_exists(encoder_type):
            prefix = self.checkpoint_filename_prefix
            return os.path.isfile("{}{}_encoder.pth".format(prefix, encoder_type))

        exists = True
        if self.disc_encoder != None:
            exists = encoder_exists("disc")
        if self.cont_encoder != None:
            exists = exists and encoder_exists("cont")
        return exists

    def save_checkpoint(self):
        if self.disc_encoder != None:
            self.save_encoder("disc")
        if self.cont_encoder != None:
            self.save_encoder("cont")

        prefix = self.checkpoint_filename_prefix
        torch.save(self.classifier, prefix + "cls.pth")
        self.logger.logger.info("=> saved checkpoint")

    def save_encoder(self, encoder_type):
        assert encoder_type in ["disc", "cont"]
        if encoder_type == "disc":
            encoder = self.disc_encoder
        else:
            encoder = self.cont_encoder

        prefix = self.checkpoint_filename_prefix
        torch.save(encoder, "{}{}_encoder.pth".format(prefix, encoder_type))
        if self.use_tgn_memory:
            torch.save(self.tgn_train_memory_backup, prefix + "tgn_memory_train.pth")
            torch.save(self.tgn_val_memory_backup, prefix + "tgn_memory_val.pth")

    def load_checkpoint(
        self, load_disc_encoder=True, load_cont_encoder=True, load_decoder=True
    ):
        if self.disc_encoder != None and load_disc_encoder:
            self.disc_encoder = self.load_encoder("disc")
        if self.cont_encoder != None and load_cont_encoder:
            self.cont_encoder = self.load_encoder("cont")

        if load_decoder:
            # Remember to initialize optimizers if the classifier is to be further optimized
            prefix = self.checkpoint_filename_prefix
            self.classifier = torch.load(
                prefix + "cls.pth",
                map_location=torch.device(self.args.device),
            )

    def load_encoder(self, encoder_type):
        assert encoder_type in ["disc", "cont"]
        if encoder_type == "disc":
            old_encoder = self.disc_encoder
        else:
            old_encoder = self.cont_encoder

        old_encoder_name = type(old_encoder).__name__

        prefix = self.checkpoint_filename_prefix
        encoder = torch.load(
            "{}{}_encoder.pth".format(prefix, encoder_type),
            map_location=torch.device(self.args.device),
        )
        if hasattr(encoder, "device"):
            encoder.device = self.args.device
            if hasattr(encoder, "memory"):  # i.e it is tgn
                encoder.memory.device = self.args.device
                encoder.message_aggregator.device = self.args.device
                encoder.memory_updater.device = self.args.device
                encoder.embedding_module.device = self.args.device
        if self.use_tgn_memory:
            self.tgn_train_memory_backup = torch.load(
                prefix + "tgn_memory_train.pth",
                map_location=torch.device(self.args.device),
            )
            self.tgn_val_memory_backup = torch.load(
                prefix + "tgn_memory_val.pth",
                map_location=torch.device(self.args.device),
            )
        self.logger.logger.info("<= loaded checkpoint {}".format(prefix))
        new_encoder_name = type(encoder).__name__

        error_msg = (
            "Loaded encoder is not correct class. Was {}, should have been {}".format(
                new_encoder_name, old_encoder_name
            )
        )
        assert old_encoder_name == new_encoder_name, error_msg

        return encoder

    def train(self):
        self.downstream = False
        self.logger = logger.Logger(
            self.args,
            self.num_classes,
            self.num_nodes,
            train_encoder=self.train_encoder,
        )
        self.report_trainable_parameters()

        def run_epochs(run_epoch, splitter, logger):
            self.tr_step = 0
            best_eval_valid = 0
            eval_valid = 0
            epochs_without_impr = 0

            for e in range(1, self.args.num_epochs + 1):
                eval_train, nodes_embs = run_epoch(splitter.train, e, "TRAIN")
                epochs_without_impr += 1
                do_eval = (
                    e >= self.args.eval_after_epochs
                    and e % self.args.eval_epoch_interval == 0
                )
                if len(splitter.val) > 0 and do_eval:
                    eval_valid, _ = run_epoch(splitter.val, e, "VALID")
                    if eval_valid > best_eval_valid:
                        best_eval_valid = eval_valid
                        epochs_without_impr = 0
                        print(
                            "### w"
                            + str(self.args.rank)
                            + ") ep "
                            + str(e)
                            + " - Best valid measure:"
                            + str(eval_valid)
                        )
                    else:
                        if epochs_without_impr > self.args.early_stop_patience:
                            print(
                                "### w"
                                + str(self.args.rank)
                                + ") ep "
                                + str(e)
                                + " - Early stop."
                            )
                            break

                if len(splitter.test) > 0 and eval_valid == best_eval_valid and do_eval:
                    self.save_checkpoint()
                    eval_test, _ = run_epoch(splitter.test, e, "TEST")

                    # if self.args.save_node_embeddings:
                    #    self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
                    # self.save_node_embs_csv(nodes_embs, self.splitter.val_idx, log_file+'_valid_nodeembs.csv.gz')
                    # self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')

        if self.train_encoder:
            run_epochs(self.run_epoch, self.splitter, self.logger)
        else:
            assert (
                self.args.temporal_granularity == "continuous"
            ), "Frozen encoder training only supported for continuous models"
            self.logger.logger.info("##### Decoder training")
            self.frozen = True
            self.load_checkpoint(load_disc_encoder=False, load_decoder=False)
            run_epochs(
                self.frozen_encoder_epoch, self.combined_splitter, self.logger
            )
            self.frozen = False  # Let it go! Let it gooo!

        if (
            hasattr(self.args, "final_epoch")
            and self.args.final_epoch
            and self.train_encoder
        ):
            # Final epoch on all edges to report accurate metrics
            self.load_checkpoint(load_decoder=True)
            if self.args.temporal_granularity != "continuous":
                run_epoch = self.run_epoch
            else:
                run_epoch = self.frozen_encoder_epoch

            run_epoch(self.final_splitter.val, self.final_epoch_id, "VALID")
            self.set_save_predictions = self.args.save_predictions
            run_epoch(self.final_splitter.test, self.final_epoch_id, "TEST")
            self.set_save_predictions = False

        self.logger.close()

        if self.args.run_downstream:
            # Downstream learning
            print("Started downstream learning")
            self.logger.logger.info("##### Downstream learning")
            self.downstream = True  # Is it pretty? No. Does it work? Yes
            self.downstream_loggers = {
                key: logger.Logger(
                    self.args, self.num_classes, self.num_nodes, classifier_name=key
                )
                for key in self.downstream_classifiers.keys()
            }
            self.load_checkpoint(load_decoder=True)
            e = 0  # epoch
            _ = self.run_downstream_epoch(self.downstream_splitter.train, e, "TRAIN")
            _ = self.run_downstream_epoch(self.downstream_splitter.val, e, "VALID")
            _ = self.run_downstream_epoch(self.downstream_splitter.test, e, "TEST")

            for logr in self.downstream_loggers.values():
                logr.close()

        self.downstream = False

    # Load model and only run on test
    def eval(self):
        self.logger = logger.Logger(
            self.args,
            self.num_classes,
            self.num_nodes,
            train_encoder=self.train_encoder,
        )

        # Load in everything
        self.load_checkpoint()

        def run_test_epoch(run_epoch, splitter, logger):
            self.set_save_predictions = self.args.save_predictions
            eval_test, _ = run_epoch(splitter.test, 1, "TEST")
            self.set_save_predictions = False

        if self.args.temporal_granularity in ["static", "discrete"]:
            run_test_epoch(self.run_epoch, self.splitter, self.logger)
        else:
            assert (
                self.args.temporal_granularity == "continuous"
            ), "Frozen encoder training only supported for continuous models"
            self.logger.logger.info("##### Decoder eval")
            self.frozen = True
            run_test_epoch(
                self.frozen_encoder_epoch, self.combined_splitter, self.logger
            )
            self.frozen = False  # Let it go! Let it gooo!

        self.logger.close()

    def _epoch_decorator(run_epoch_func):
        @functools.wraps(run_epoch_func)
        def wrapper(*args, **kwards):
            self = args[0]
            split = args[1]
            epoch = args[2]
            set_name = args[3]

            if self.use_tgn_memory:
                self.prepare_tgn_memory(set_name)

            log_interval = 999
            if set_name == "TEST":
                log_interval = 1

            # Epoch start logger(s)
            if not self.downstream:
                self.logger.log_epoch_start(
                    epoch, len(split), set_name, minibatch_log_interval=log_interval
                )
            else:
                # Using variable name logr instead of logger to avoid overwriting import logger
                for logr in self.downstream_loggers.values():
                    logr.log_epoch_start(
                        epoch, len(split), set_name, minibatch_log_interval=log_interval
                    )

            # Run epoch
            nodes_embs = run_epoch_func(*args, **kwards)

            if self.use_tgn_memory:
                self.backup_tgn_memory(set_name)

            # Epoch done logger(s)
            if not self.downstream:
                eval_measure = self.logger.log_epoch_done()
            else:
                for logr in self.downstream_loggers.values():
                    logr.log_epoch_done()
                eval_measure = None  # Doesn't matter here
            return eval_measure, nodes_embs

        return wrapper

    @_epoch_decorator
    def run_downstream_epoch(self, split, epoch, set_name):
        raise NotImplementedError

        self.encoder = self.encoder.eval()
        torch.set_grad_enabled(False)

        for s in split:
            encode_sample, test_sample = s

            # Encoder
            es = self.prepare_sample(encode_sample, self.args.temporal_granularity)
            nodes_embs = self.encode(es, set_name)
            downstream_loss = torch.tensor(0.0)  # No loss since we only encode

            # Downstream
            s = self.prepare_sample(test_sample, "static", only_label_sp=True)
            predictions_dict, probs_dict = self.predict_downstream(
                nodes_embs, s.label_sp["idx"], s.label_sp["vals"], set_name
            )
            for classifier_name in predictions_dict:
                log = self.downstream_loggers[classifier_name]
                if set_name in ["TEST", "VALID"] and self.args.task == "link_pred":
                    log.log_minibatch(
                        downstream_loss.detach(),
                        predictions_dict[classifier_name],
                        probs_dict[classifier_name],
                        s.label_sp["vals"],
                        adj=s.label_sp["idx"],
                        prev_adj=s.prev_adj["idx"],
                    )
                else:
                    log.log_minibatch(
                        downstream_loss.detach(),
                        predictions_dict[classifier_name],
                        probs_dict[classifier_name],
                        s.label_sp["vals"],
                    )

        self.encoder = self.encoder.train()
        torch.set_grad_enabled(True)

        return nodes_embs

    @_epoch_decorator
    def frozen_encoder_epoch(self, split, epoch, set_name):

        if set_name == "TRAIN":
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        # Freeze encoder
        self.cont_encoder = self.cont_encoder.eval()
        for param in self.cont_encoder.parameters():
            param.requires_grad = False

        # Cache encoder embeddings for each snapshot
        to_cache = len(self.embedding_cache[set_name]) == 0
        if to_cache:
            self.logger.logger.info("Cache empty, encoding to fill cache")
        else:
            self.logger.logger.info("Using cache")

        # Epoch
        i = 0
        for s in split:
            encode_sample, test_sample = s

            i = i + 1
            if to_cache:
                # Update cache
                es = self.prepare_sample(encode_sample, self.args.temporal_granularity)
                nodes_embs = self.encode(es, set_name)
                self.embedding_cache[set_name][i] = nodes_embs
            else:
                # Use cache
                nodes_embs = self.embedding_cache[set_name][i]
            # assert nodes_embs.isnan().any() == False, 'A node embedding is nan'

            # Node embs is occasionally nan. Sets nan to zero
            nodes_embs[nodes_embs.isnan()] = 0

            s = self.prepare_sample(test_sample, "static", only_label_sp=True)
            # Decoder
            predictions = self.predict(nodes_embs, s.label_sp)

            loss = self.comp_loss(predictions, s.label_sp["vals"])
            # print("loss:", loss)
            # assert math.isnan(loss.item()) == False, 'Loss is nan'
            probs = torch.softmax(predictions, dim=1)[:, 1]

            if self.set_save_predictions:
                if (
                    hasattr(self.args, "custom_labeler")
                    and self.args.custom_labeler == True
                ):
                    settype = self.tasker.custom_labeler.settype
                    prefix = "{}_{}".format(self.prediction_filename_prefix, settype)
                    u.save_predictions(
                        probs,
                        s.label_sp["idx"],
                        s.label_sp["vals"],
                        i,
                        prefix,
                        self.dataset,
                    )
                else:
                    u.save_predictions(
                        probs,
                        s.label_sp["idx"],
                        s.label_sp["vals"],
                        i,
                        self.prediction_filename_prefix,
                        self.dataset,
                    )

            if set_name in ["TEST", "VALID"] and self.args.task == "link_pred":
                self.logger.log_minibatch(
                    loss.detach(),
                    predictions.detach().cpu(),
                    probs.detach().cpu(),
                    s.label_sp["vals"],
                    adj=s.label_sp["idx"],
                    prev_adj=s.prev_adj["idx"],
                )
            else:
                self.logger.log_minibatch(
                    loss.detach(),
                    predictions.detach().cpu(),
                    probs.detach().cpu(),
                    s.label_sp["vals"],
                )
            if set_name == "TRAIN":
                self.optim_step_decoder(loss)

        self.cont_encoder = self.cont_encoder.train()
        torch.set_grad_enabled(True)

        return nodes_embs

    @_epoch_decorator
    def run_epoch(self, split, epoch, set_name):
        snapshot_free = (
            self.args.temporal_granularity == "continuous" or self.args.model == "seal"
        )

        if set_name == "TRAIN":
            if self.disc_encoder != None:
                self.disc_encoder.train()
            if self.cont_encoder != None:
                self.cont_encoder.train()
            # If the cls is using dropout also call cls.train here. However we currently don't.
            torch.set_grad_enabled(True)
        else:
            if self.disc_encoder != None:
                self.disc_encoder.eval()
            if self.cont_encoder != None:
                self.cont_encoder.eval()
            torch.set_grad_enabled(False)

        i = 0
        for s in split:
            # split is a data_split class and this calls the __get_item__ function
            # s = sample
            # for key in s.keys():
            #   print(key, u.naturalsize(u.get_memory_size(s[key])))
            # Reshapes and sends the tensors to device
            s = self.prepare_sample(s, self.args.temporal_granularity)
            i = i + 1
            # print("ss", i, 'set name', set_name)

            if not snapshot_free:  # Snapshots, i.e. the Static, Discrete
                nodes_embs = self.encode(s, set_name)  # Encoder
                predictions = self.predict(nodes_embs, s.label_sp)  # Decoder

                loss = self.comp_loss(predictions, s.label_sp["vals"])
                probs = torch.softmax(predictions, dim=1)[:, 1]

                if self.set_save_predictions:
                    settype = self.tasker.custom_labeler.settype
                    prefix = "{}_{}".format(self.prediction_filename_prefix, settype)
                    u.save_predictions(
                        probs,
                        s.label_sp["idx"],
                        s.label_sp["vals"],
                        i,
                        prefix,
                        self.dataset,
                    )

                if set_name in ["TEST", "VALID"] and self.args.task == "link_pred":
                    self.logger.log_minibatch(
                        loss.detach(),
                        predictions.detach().cpu(),
                        probs.detach().cpu(),
                        s.label_sp["vals"],
                        adj=s.label_sp["idx"],
                        prev_adj=s.prev_adj["idx"],
                    )
                else:
                    self.logger.log_minibatch(
                        loss.detach(),
                        predictions.detach().cpu(),
                        probs.detach().cpu(),
                        s.label_sp["vals"],
                    )
                if set_name == "TRAIN" and not self.downstream:
                    self.optim_step(loss)  # Only for DGNN training
            else:  # Edge based training - including continuous
                if self.args.model == "seal":
                    nodes_embs = self.predict_seal(s, set_name)
                else:
                    nodes_embs = self.predict_continuous(
                        s.hist_adj,
                        s.hist_time,
                        s.hist_ndFeats,
                        s.hist_node_mask,
                        set_name,
                    )
                # Logging done internally in continuous training

        if self.disc_encoder != None:
            self.disc_encoder.train()
        if self.cont_encoder != None:
            self.cont_encoder.train()
        torch.set_grad_enabled(True)

        return nodes_embs

    def encode(self, sample, set_name, temporal_granularity=None):
        if temporal_granularity == None:
            temporal_granularity = self.args.temporal_granularity

        if temporal_granularity != "continuous":
            nodes_embs = self.disc_encoder(
                sample.hist_adj,
                sample.hist_ndFeats,
                sample.hist_vals,
                sample.hist_node_mask,
            )
        else:  # If snapshot based and continuous, used for downstream learning.
            nodes_embs = self.predict_continuous(
                sample.hist_adj,
                sample.hist_time,
                sample.hist_ndFeats,
                sample.hist_node_mask,
                set_name,
            )
        return nodes_embs

    def predict(self, nodes_embs, label_sp):
        node_indices = label_sp["idx"]
        if self.has_time_query:
            time = label_sp["time"]
            edge_type = label_sp["type"]
            time_feats = label_sp["time_feats"]
        batch_size = self.args.decoder_batch_size
        gather_predictions = []
        for i in range(1 + (node_indices.size(1) // batch_size)):
            b_start = i * batch_size
            b_end = (i + 1) * batch_size
            links = node_indices[:, b_start:b_end]
            x = self.gather_node_embs(nodes_embs, links)

            if self.has_time_query:
                t = time[b_start:b_end].unsqueeze(1)
                e_type = edge_type[b_start:b_end, :]
                t_feats = time_feats[b_start:b_end, :]
                predictions = self.classifier(x, t, e_type, t_feats)
            else:
                predictions = self.classifier(x)

            gather_predictions.append(predictions)
        gather_predictions = torch.cat(gather_predictions, dim=0)

        return gather_predictions

    def gather_node_embs(self, nodes_embs, links):
        cls_input = []

        for node_set in links:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input, dim=1)

    def predict_continuous(self, hist_adj, hist_time, hist_ndFeats, mask, set_name):
        batch_size = self.args.continuous_batch_size
        assert len(hist_adj) == len(hist_time) == len(hist_ndFeats) == 1

        # Some Torch to numpy and GPU to CPU ping pong caused by TGAT taking numpy as input
        # Luckily this happens only 3 times per epoch.
        adj = hist_adj[0]
        times = hist_time[0].cpu().numpy()
        assert len(adj) == 2

        src_idx_l = adj[0].cpu().numpy()
        target_idx_l = adj[1].cpu().numpy()
        num_nodes = len(np.unique(src_idx_l))
        num_edges = len(np.atleast_1d(src_idx_l))

        if num_edges <= 1:  # The rest of the function assumes multiple edges.
            # Ignore this one edge and simply return previous embeddings. This may happen on sparse datasets with small snapshots (the beginning of UC is an example)
            nodes_embs = self.cont_encoder.node_embed.detach()
            assert (
                len(nodes_embs) == self.num_nodes
            ), "Node embeddings need to include all nodes"
            return nodes_embs
        num_batches = num_edges // batch_size

        # The below line is potentially great for datasets which have no edge features. But if the initialization is dictated by edge features (as in TGAT), then we'll have to do something different here.
        # self.cont_encoder.update_node_features(hist_ndFeats[0].to_dense().cpu().numpy())

        # Combining source and target since TGAT is based on node embeddings and since our networks are treated as undirected.
        # The ngh_finder retrieves a sample of edges per node which are used for the embedding.
        # nodes = torch.stack([src_idx_l,target_idx_l], dim=1).flatten()
        # times = torch.stack([times, times], dim=1).flatten()

        for i in range(1 + num_batches):
            # print("Batch {}/{}".format(i+1, num_batches+1))
            src_batch, target_batch, times_batch, edge_idxs = self.get_continuous_batch(
                src_idx_l, target_idx_l, times, i, batch_size
            )
            # print("continuous batch times", times_batch.min(), times_batch.max())

            size = len(src_batch)
            if size <= 1:
                continue  # TGAT breaks down if the batch only contains one edge.
            # nembs_batch = self.cont_encoder(src_batch, #contrast used instead
            #                 times_batch,
            #                 hist_ndFeats[0],
            #                 mask)
            target_l_fake = np.random.randint(0, num_nodes, size)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=self.args.device)
                neg_label = torch.zeros(
                    size, dtype=torch.float, device=self.args.device
                )

            self.opt_cont_encoder.zero_grad()
            pos_prob, neg_prob = self.cont_encoder.contrast(
                src_batch, target_batch, target_l_fake, times_batch, edge_idxs
            )

            if not self.downstream and not self.frozen:
                # If we're downstream or frozen we just want to encode
                inter_loss = self.interpolation_loss(pos_prob, pos_label)
                inter_loss += self.interpolation_loss(neg_prob, neg_label)

                if set_name == "TRAIN":
                    inter_loss.backward()
                    self.opt_cont_encoder.step()
                    if self.use_tgn_memory:
                        self.cont_encoder.memory.detach_memory()

                with torch.no_grad():
                    self.cont_encoder = self.cont_encoder.eval()
                    probs = torch.tensor(
                        np.concatenate(
                            [
                                (pos_prob).cpu().detach().numpy(),
                                (neg_prob).cpu().detach().numpy(),
                            ]
                        )
                    )
                    predictions = torch.stack((probs, 1 - probs), dim=1)
                    true_label = torch.tensor(
                        np.concatenate([np.ones(size), np.zeros(size)])
                    )
                    self.logger.log_minibatch(
                        inter_loss.detach(),
                        predictions.detach().cpu(),
                        probs.detach().cpu(),
                        true_label,
                        calc_lp_metrics=False,
                    )

        # Detach breaks the link between the models so the continuous model and classifier are trained separately.
        nodes_embs = self.cont_encoder.node_embed.detach()
        assert (
            len(nodes_embs) == self.num_nodes
        ), "Node embeddings need to include all nodes"
        return nodes_embs

    def predict_downstream(self, nodes_embs, node_index, true_classes, set_name):
        # If training we can only fit once, so no batching.
        # Since we train using negative samples the data should be small enough to be handled in one go.
        nodes_embs = nodes_embs.cpu()
        node_index = node_index.cpu()
        true_classes = true_classes.cpu()

        if set_name == "TRAIN":
            predictions_dict, probs_dict = self.predict_downstream_batch(
                nodes_embs, node_index, true_classes, set_name
            )
            return predictions_dict, probs_dict

        # Batching for validation and test
        def get_batch(i, batch_size, nodes_embs, node_index, true_classes, set_name):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            node_index_batch = node_index[:, batch_start:batch_end]
            true_classes_batch = true_classes[batch_start:batch_end]
            predictions_dict, probs_dict = self.predict_downstream_batch(
                nodes_embs, node_index_batch, true_classes_batch, set_name
            )
            return predictions_dict, probs_dict

        batch_size = self.args.decoder_batch_size
        # TODO Parallel seems to make it slower, consider investigating in the future, maybe different batch size
        # parallel = Parallel(n_jobs=-1)
        # gather_predictions = parallel(delayed(get_batch)(
        #    i, batch_size, nodes_embs, node_index, true_classes, set_name)
        #    for i in range(1 +(node_index.size(1)//batch_size))
        # )
        gather_predictions = [
            get_batch(i, batch_size, nodes_embs, node_index, true_classes, set_name)
            for i in range(1 + (node_index.size(1) // batch_size))
        ]

        # Prepare predictions for concatenation
        predictions_dict_list = {}
        probs_dict_list = {}
        for predictions_dict, probs_dict in gather_predictions:
            for classifier_name in predictions_dict:
                if classifier_name in predictions_dict_list.keys():
                    predictions_dict_list[classifier_name].append(
                        predictions_dict[classifier_name]
                    )
                    probs_dict_list[classifier_name].append(probs_dict[classifier_name])
                else:
                    predictions_dict_list[classifier_name] = [
                        predictions_dict[classifier_name]
                    ]
                    probs_dict_list[classifier_name] = [probs_dict[classifier_name]]

        # Concatenate batched predictions
        predictions_dict = {}
        probs_dict = {}
        for classifier_name in predictions_dict_list:
            predictions = torch.cat(predictions_dict_list[classifier_name], dim=0)
            probs = torch.cat(probs_dict_list[classifier_name], dim=0)

            predictions_dict[classifier_name] = predictions
            probs_dict[classifier_name] = probs
        return predictions_dict, probs_dict

    def predict_downstream_batch(self, nodes_embs, node_index, true_classes, set_name):
        embedding_size = nodes_embs[1].size()[0]
        n1, n2 = torch.split(
            self.gather_node_embs(nodes_embs, node_index), embedding_size, dim=1
        )
        X = np.array(torch.cat([n1, n2, np.multiply(n1, n2)], dim=1))
        y = np.array(true_classes)

        # A rare problem where some embeddings on some snapshots cause X to include nan values
        if not np.isfinite(X).all():
            for logr in self.downstream_loggers.values():
                logr.logger.warning(
                    "nan/inf/-inf observed in downstream X. Setting values to default valid numbers"
                )
            X = np.nan_to_num(X)

        predictions_dict = {}
        probs_dict = {}

        for key, classifier in self.downstream_classifiers.items():
            if set_name == "TRAIN":
                classifier.fit(X, y)
            predictions = classifier.predict_proba(X)
            probs = predictions[:, 1]
            self.downstream_classifiers[key] = classifier
            predictions_dict[key] = torch.tensor(predictions)
            probs_dict[key] = torch.tensor(probs)
        return predictions_dict, probs_dict

    def get_continuous_batch(self, src_l, target_l, times_l, i, batch_size):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        src_batch = src_l[start_idx:end_idx]
        target_batch = target_l[start_idx:end_idx]
        times_batch = times_l[start_idx:end_idx]
        edge_idxs = np.arange(start_idx, start_idx + len(src_batch))
        # print("batch", i, "src", src_l.shape, "sidx", start_idx, "eidx", end_idx, "src batch", src_batch.shape)
        return src_batch, target_batch, times_batch, edge_idxs

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            if self.disc_encoder != None and self.args.model != "random":
                self.opt_encoder.step()
                self.opt_encoder.zero_grad()
            self.opt_decoder.step()
            self.opt_decoder.zero_grad()

    def optim_step_decoder(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.opt_decoder.step()
            self.opt_decoder.zero_grad()

    def prepare_sample(
        self, sample, temporal_granularity="static", only_label_sp=False
    ):
        sample = u.Namespace(sample)
        sample.hist_vals, sample.hist_time = [], []
        if self.args.model == "seal":
            # For SEAL we want to pack the data edge by edge into a dataloader (yes a second one)
            seal_dataset = SEALDataset(
                self.args,
                self.args.gcn_parameters["hops"],
                self.tasker.prepare_node_feats(sample.hist_ndFeats[0]).to_dense(),
                sample.label_exist["idx"].squeeze(),
                sample.label_non_exist["idx"].squeeze(),
                sample.hist_adj[0]["idx"].squeeze(),
            )
            seal_loader = pygeomDataLoader(seal_dataset, batch_size=32)
            return seal_loader  # Returns a dataloader instead of a sample if it is SEAL
        else:
            # For the static and continuous case there will be only one iteration
            for i, adj in enumerate(sample.hist_adj):
                # Prepares an edge index (edge list) as expected by PyTorch Geometric
                # Squeeze removes dimensions of size 1
                vals = adj["vals"].squeeze().t()
                sample.hist_vals.append(vals.to(self.args.device))
                if temporal_granularity == "continuous":
                    hist_time = adj["time"].squeeze().t()
                    sample.hist_time.append(hist_time.to(self.args.device))

                if hasattr(self.args, "pygeom") and self.args.pygeom == False:
                    # Only used for the original implementation of EGCN
                    adj_idx = u.sparse_prepare_tensor(adj, torch_size=[self.num_nodes])
                else:
                    adj_idx = adj["idx"].squeeze().t()
                sample.hist_adj[i] = adj_idx.to(self.args.device)

                if not only_label_sp:
                    # Created some problems for reddit_tgn, we don't use this there anyways.
                    if self.disc_tasker != None:
                        nodes = self.disc_tasker.prepare_node_feats(
                            sample.hist_ndFeats[i]
                        )
                    else:
                        nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats[i])

                    sample.hist_ndFeats[i] = nodes.to(self.args.device)
                    hist_node_mask = sample.hist_node_mask[i]
                    # transposed to have same dimensions as scorer
                    sample.hist_node_mask[i] = hist_node_mask.to(self.args.device).t()

        label_sp = self.ignore_batch_dim(sample.label_sp)
        if self.has_time_query:
            label_sp["time"] = label_sp["time"].squeeze().to(self.args.device)
            label_sp["type"] = label_sp["type"].squeeze().to(self.args.device)
            label_sp["time_feats"] = (
                label_sp["time_feats"].squeeze().to(self.args.device)
            )

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp["idx"] = label_sp["idx"].to(self.args.device).t()
        else:
            label_sp["idx"] = label_sp["idx"].to(self.args.device)

        label_sp["vals"] = label_sp["vals"].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj["idx"] = adj["idx"][0]
        adj["vals"] = adj["vals"][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

            csv_node_embs.append(
                torch.cat((orig_ID, nodes_embs[node_id].double())).detach().numpy()
            )

        pd.DataFrame(np.array(csv_node_embs)).to_csv(
            file_name, header=None, index=None, compression="gzip"
        )
        print("Node embs saved in", file_name)

    def prepare_tgn_memory(self, set_name):
        if set_name == "TRAIN":
            self.logger.logger.info("init memory")
            self.cont_encoder.memory.__init_memory__()
        elif set_name == "VALID":
            self.logger.logger.info("restore training memory")
            assert self.tgn_train_memory_backup is not None
            self.cont_encoder.memory.restore_memory(self.tgn_train_memory_backup)
        elif set_name == "TEST":
            self.logger.logger.info("restore validation memory")
            assert self.tgn_val_memory_backup is not None
            self.cont_encoder.memory.restore_memory(self.tgn_val_memory_backup)

    def backup_tgn_memory(self, set_name):
        if set_name == "TRAIN":
            print("save train memory")
            self.tgn_train_memory_backup = self.cont_encoder.memory.backup_memory()
            assert self.tgn_train_memory_backup is not None
        elif set_name == "VALID":
            print("save validation memory")
            self.tgn_val_memory_backup = self.cont_encoder.memory.backup_memory()
            assert self.tgn_val_memory_backup is not None
