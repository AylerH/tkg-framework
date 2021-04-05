import torch

import time
import os
import argparse

from typing import Dict, List
from collections import defaultdict

from tkge.task.task import Task
from tkge.data.dataset import DatasetProcessor, SplitDataset
from tkge.train.sampling import NegativeSampler, NonNegativeSampler
from tkge.train.regularization import Regularizer, InplaceRegularizer
from tkge.train.optim import get_optimizer, get_scheduler
from tkge.common.config import Config
from tkge.models.model import BaseModel
from tkge.models.pipeline_model import TransSimpleModel
from tkge.models.loss import Loss
from tkge.eval.metrics import Evaluation


class Trainer():
    def __init__(self, config: Config):
        self.dataset: DatasetProcessor = self.config.get("dataset.name")
        self.train_loader: torch.utils.data.DataLoader = None
        self.valid_loader: torch.utils.data.DataLoader = None
        # self.test_loader = None
        self.sampler: NegativeSampler = None
        self.model: BaseModel = None
        self.loss: Loss = None
        self.optimizer: torch.optim.optimizer.Optimizer = None
        self.lr_scheduler = None
        self.evaluation: Evaluation = None

        self.train_bs = self.config.get("train.batch_size")
        self.valid_bs = self.config.get("train.valid.batch_size")
        self.train_sub_bs = self.config.get("train.subbatch_size") if self.config.get(
            "train.subbatch_size") else self.train_bs
        self.valid_sub_bs = self.config.get("train.valid.subbatch_size") if self.config.get(
            "train.valid.subbatch_size") else self.valid_bs

        self.datatype = (['timestamp_id'] if self.config.get("dataset.temporal.index") else []) + (
            ['timestamp_float'] if self.config.get("dataset.temporal.float") else [])

        # TODO(gengyuan): passed to all modules
        self.device = self.config.get("task.device")

        self._prepare()

    def _prepare(self):
        self.config.log(f"Preparing datasets {self.dataset} in folder {self.config.get('dataset.folder')}...")
        self.dataset = DatasetProcessor.create(config=self.config)
        self.dataset.info()

        self.config.log(f"Loading training split data for loading")
        # TODO(gengyuan) load params
        self.train_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("train"), self.datatype),
            shuffle=True,
            batch_size=self.train_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.valid_loader = torch.utils.data.DataLoader(
            SplitDataset(self.dataset.get("test"), self.datatype + ['timestamp_id']),
            shuffle=False,
            batch_size=self.valid_bs,
            num_workers=self.config.get("train.loader.num_workers"),
            pin_memory=self.config.get("train.loader.pin_memory"),
            drop_last=self.config.get("train.loader.drop_last"),
            timeout=self.config.get("train.loader.timeout")
        )

        self.config.log(f"Initializing negative sampling")
        self.sampler = NegativeSampler.create(config=self.config, dataset=self.dataset)
        self.onevsall_sampler = NonNegativeSampler(config=self.config, dataset=self.dataset, as_matrix=True)

        self.config.log(f"Creating model {self.config.get('model.type')}")
        self.model = BaseModel.create(config=self.config, dataset=self.dataset)
        self.model.to(self.device)

        self.config.log(f"Initializing loss function")
        self.loss = Loss.create(config=self.config)

        self.config.log(f"Initializing optimizer")
        optimizer_type = self.config.get("train.optimizer.type")
        optimizer_args = self.config.get("train.optimizer.args")
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_type, optimizer_args)

        self.config.log(f"Initializing lr scheduler")
        if self.config.get("train.lr_scheduler"):
            scheduler_type = self.config.get("train.lr_scheduler.type")
            scheduler_args = self.config.get("train.lr_scheduler.args")
            self.lr_scheduler = get_scheduler(self.optimizer, scheduler_type, scheduler_args)

        self.config.log(f"Initializing regularizer")
        self.regularizer = dict()
        self.inplace_regularizer = dict()

        if self.config.get("train.regularizer"):
            for name in self.config.get("train.regularizer"):
                self.regularizer[name] = Regularizer.create(self.config, name)

        if self.config.get("train.inplace_regularizer"):
            for name in self.config.get("train.inplace_regularizer"):
                self.inplace_regularizer[name] = InplaceRegularizer.create(self.config, name)

        self.config.log(f"Initializing evaluation")
        self.evaluation = Evaluation(config=self.config, dataset=self.dataset)

        # validity checks and warnings
        if self.train_sub_bs >= self.train_bs or self.train_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.sub_batch_size={self.train_sub_bs} is greater or equal to "
                            f"train.batch_size={self.train_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.train_sub_bs = self.train_bs

        if self.valid_sub_bs >= self.valid_bs or self.valid_sub_bs < 1:
            # TODO(max) improve logging with different hierarchies/labels, i.e. merge with branch advannced_log_and_ckpt_management
            self.config.log(f"Specified train.valid.sub_batch_size={self.valid_sub_bs} is greater or equal to "
                            f"train.valid.batch_size={self.valid_bs} or smaller than 1, so use no sub batches. "
                            f"Device(s) may run out of memory.", level="warning")
            self.valid_sub_bs = self.valid_bs

    def run_epoch(self):
        self.model.train()

        # TODO early stopping conditions
        # 1. metrics 变化小
        # 2. epoch
        # 3. valid koss
        total_epoch_loss = 0.0
        train_size = self.dataset.train_size

        start_time = time.time()

        for pos_batch in self.train_loader:
            self.optimizer.zero_grad()

            batch_loss = 0.

            # may be smaller than the specified batch size in last iteration
            bs = pos_batch.size(0)

            for start in range(0, bs, self.train_sub_bs):
                stop = min(start + self.train_sub_bs, bs)
                pos_subbatch = pos_batch[start:stop]
                subbatch_loss, subbatch_factors = self._subbatch_forward(pos_subbatch)

                batch_loss += subbatch_loss

            batch_loss.backward()
            self.optimizer.step()

            total_epoch_loss += batch_loss.cpu().item()

            if subbatch_factors:
                for name, tensors in subbatch_factors.items():
                    if name not in self.inplace_regularizer:
                        continue

                    if not isinstance(tensors, (tuple, list)):
                        tensors = [tensors]

                    self.inplace_regularizer[name](tensors)

            # empty caches
            # del samples, labels, scores, factors
            # if self.device=="cuda":
            #     torch.cuda.empty_cache()

        stop_time = time.time()
        avg_loss = total_epoch_loss / train_size

        return avg_loss, stop_time - start_time

    def _subbatch_forward(self, pos_subbatch):
        sample_target = self.config.get("negative_sampling.target")
        samples, labels = self.sampler.sample(pos_subbatch, sample_target)

        samples = samples.to(self.device)
        labels = labels.to(self.device)

        scores, factors = self.model.fit(samples)

        assert scores.size(0) == labels.size(
            0), f"Score's size {scores.shape} should match label's size {labels.shape}"
        loss = self.loss(scores, labels)

        assert not (factors and set(factors.keys()) - (set(self.regularizer) | set(
            self.inplace_regularizer))), f"Regularizer name defined in model {set(factors.keys())} should correspond to that in config file"

        if factors:
            for name, tensors in factors.items():
                if name not in self.regularizer:
                    continue

                if not isinstance(tensors, (tuple, list)):
                    tensors = [tensors]

                reg_loss = self.regularizer[name](tensors)
                loss += reg_loss

        return loss, factors