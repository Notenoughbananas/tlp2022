from dataclasses import dataclass
import utils as u
from datareaders.datareader import Datareader

# taskers
import link_pred_tasker as lpt

import splitter as sp
import Cross_Entropy as ce

import trainer as tr

# Logic for setting up an experiment. Ultimately calls trainer.py
class Experiment:
    def __init__(
        self,
        args,
        build_tasker,
        build_gcn,
        build_classifier,
        build_custom_labeler,
        build_tlp_labeler,
        eval_only=False,
    ):
        self.args = args
        self.eval_only = eval_only
        self.container = None
        self.build_tasker = build_tasker
        self.build_gcn = build_gcn
        self.build_classifier = build_classifier
        self.build_custom_labeler = build_custom_labeler
        self.build_tlp_labeler = build_tlp_labeler

        self.has_time_query = (
            hasattr(args, "has_time_query") and args.has_time_query == True
        )
        self.has_custom_lb = (
            hasattr(args, "custom_labeler") and args.custom_labeler == True
        )

        self.prepare_container()

    ###### Part 1: Prepare container
    def prepare_container(self):
        if self.args.temporal_granularity in ["static", "discrete"]:
            self.container = self._prepare_discrete_data_training()
        elif ( self.args.temporal_granularity == "continuous" ):
            self.container = self._prepare_continuous_data_training()
        else:
            print("What happened?!")
            raise TypeError

    def get_labelers(self, cont_dataset, disc_dataset):
        # build the tasker
        if self.has_time_query:
            tlp_labeler = self.build_tlp_labeler(self.args, cont_dataset)
        else:
            tlp_labeler = None

        if self.has_custom_lb and self.eval_only:
            custom_labeler = self.build_custom_labeler(self.args, cont_dataset)
        else:
            custom_labeler = None

        return tlp_labeler, custom_labeler

    def _prepare_discrete_data_training(self):
        args = self.args
        cont_dataset, dataset = Datareader(args).dataset

        tlp_labeler, custom_labeler = self.get_labelers(cont_dataset, dataset)

        tasker = self.build_tasker(
            args,
            dataset,
            args.temporal_granularity,
            custom_labeler=custom_labeler,
            tlp_labeler=tlp_labeler,
        )
        # build the splitters
        splitter = sp.splitter(
            args,
            tasker,
            args.temporal_granularity,
            all_in_one_snapshot=False,
            eval_all_edges=args.full_test_during_run,
            eval_only_last=self.eval_only,
        )

        # downstream_splitter = sp.splitter(
        #    args, tasker, args.temporal_granularity, downstream=True
        # )
        # Used for final metric reporting
        final_splitter = sp.splitter(
            args,
            tasker,
            args.temporal_granularity,
            all_in_one_snapshot=False,
            eval_all_edges=True,
            eval_only_last=self.eval_only,
        )

        if not args.heuristic:
            return Disc_container(self.args, dataset, tasker, splitter, final_splitter)
        else:
            # Build hist_splitter
            temp = args.num_hist_steps  # Ensure that we start with the same offset.
            args.num_hist_steps = 1
            hist_splitter = sp.splitter(
                args,
                tasker,
                args.temporal_granularity,
                all_in_one_snapshot=False,
                eval_all_edges=True,
                train_all_edges=True,
            )
            args.num_hist_steps = temp
            return Disc_container(
                self.args,
                dataset,
                tasker,
                splitter,
                final_splitter,
                hist_splitter=hist_splitter,
            )

    def _prepare_continuous_data_training(self):
        args = self.args
        cont_dataset, disc_dataset = Datareader(args).dataset
        assert disc_dataset.max_time == cont_dataset.max_time

        tlp_labeler, custom_labeler = self.get_labelers(cont_dataset, disc_dataset)

        # build the taskers
        cont_tasker = self.build_tasker(
            args,
            cont_dataset,
            "continuous",
            custom_labeler=custom_labeler,
            tlp_labeler=tlp_labeler,
        )
        disc_tasker = self.build_tasker(
            args,
            disc_dataset,
            "discrete",
            custom_labeler=custom_labeler,
            tlp_labeler=tlp_labeler,
        )

        # build the splitters
        # all_in_one_snapshot = args.temporal_granularity == 'continuous' or args.model == 'seal'
        # eval_all_edges = not all_in_one_snapshot #args.model != 'seal'
        cont_splitter = sp.splitter(
            args,
            cont_tasker,
            "continuous",
            all_in_one_snapshot=True,
            eval_all_edges=False,
        )
        # build the tasker. This needs to be done in a static way because it is for testing.

        # Downstream splitter might need work. Requires disc_tasker. Also verify whether splitter delivers samples from correct taskers.
        # downstream_splitter = sp.splitter(
        #    args, disc_tasker, args.temporal_granularity, downstream=True
        # )
        combined_splitter = sp.splitter(
            args,
            cont_tasker,
            "static",
            disc_tasker=disc_tasker,
            frozen_encoder=True,
            eval_all_edges=args.full_test_during_run,
            eval_only_last=self.eval_only,
        )  # The tasker is passed here to enable encoding for the continuous model

        final_splitter = sp.splitter(
            args,
            cont_tasker,
            "static",
            disc_tasker=disc_tasker,
            frozen_encoder=True,
            eval_all_edges=True,
            eval_only_last=self.eval_only,
        )
        return Cont_container(
            self.args,
            disc_dataset,
            disc_tasker,
            cont_dataset,
            cont_tasker,
            cont_splitter,
            combined_splitter,
            final_splitter,
        )

    ##### Part 2: Run container
    def build_trainer(self):
        assert self.container != None
        container = self.container
        args = self.args

        if args.heuristic:
            trainer = self._get_trainer_heuristic()
        else:
            self.loss = ce.Cross_Entropy(args, container.disc_dataset).to(args.device)
            if args.temporal_granularity in ["static", "discrete"]:
                trainer = self._get_trainer_discrete_data_gcn()
            elif args.temporal_granularity == "continuous":
                trainer = self._get_trainer_continuous_data_gcn()
        return trainer

    def _get_trainer_discrete_data_gcn(self):
        tasker = self.container.disc_tasker
        dataset = self.container.disc_dataset
        splitter = self.container.disc_splitter
        final_splitter = self.container.final_splitter
        feats_per_node = tasker.feats_per_node

        # build the models
        gcn, args = self.build_gcn(self.args, tasker, dataset, splitter, feats_per_node)
        classifier = self.build_classifier(args, tasker)

        trainer = tr.Trainer(
            args,
            splitter=splitter,
            final_splitter=final_splitter,
            disc_encoder=gcn,
            classifier=classifier,
            comp_loss=self.loss,
            dataset=dataset,
        )
        return trainer

    def _get_trainer_continuous_data_gcn(self):
        args = self.args
        disc_tasker = self.container.disc_tasker
        cont_tasker = self.container.cont_tasker
        cont_dataset = self.container.cont_dataset
        cont_splitter = self.container.cont_splitter
        final_splitter = self.container.final_splitter

        feats_per_node = cont_tasker.feats_per_node
        gcn, args = self.build_gcn(
            args, cont_tasker, cont_dataset, cont_splitter, feats_per_node
        )

        self._pretrain_continuous_gcn(gcn)
        classifier = self.build_classifier(args, disc_tasker)

        # Decoder training
        decoder_trainer = tr.Trainer(
            args,
            splitter=cont_splitter,
            combined_splitter=combined_splitter,
            final_splitter=final_splitter,
            cont_encoder=gcn,
            classifier=classifier,
            comp_loss=self.loss,
            dataset=cont_dataset,
            train_encoder=False,
        )
        return decoder_trainer

    def _pretrain_continuous_gcn(self, gcn):
        # Continuous encoder training
        args = self.args
        cont_dataset = self.container.cont_dataset
        cont_splitter = self.container.cont_splitter
        final_splitter = self.container.final_splitter

        decoder_num_epochs = args.num_epochs
        args.num_epochs = args.num_epochs_continuous
        trainer = tr.Trainer(
            args,
            splitter=cont_splitter,
            combined_splitter=None,
            final_splitter=final_splitter,
            cont_encoder=gcn,
            classifier=None,
            comp_loss=self.loss,
            dataset=cont_dataset,
            train_encoder=True,
        )

        assert trainer.disc_encoder == None
        force_encode = hasattr(args, "force_encode") and args.force_encode == True
        if force_encode or not trainer.checkpoint_exists():
            print(
                "Not found continuous checkpoint for combined training. Starting pre-training of continuous encoder."
            )
            trainer.train()
        else:
            print("Found continuous checkpoint for combined training.")

        args.num_epochs = decoder_num_epochs

@dataclass
class Disc_container:
    args: u.Namespace
    disc_dataset: "typing.Any"
    disc_tasker: "typing.Any"
    disc_splitter: "typing.Any"
    final_splitter: "typing.Any"
    hist_splitter: "typing.Any" = None


@dataclass
class Cont_container:
    args: u.Namespace
    disc_dataset: "typing.Any"
    disc_tasker: "typing.Any"
    cont_dataset: "typing.Any"
    cont_tasker: "typing.Any"
    cont_splitter: "typing.Any"
    combined_splitter: "typing.Any"
    final_splitter: "typing.Any"
