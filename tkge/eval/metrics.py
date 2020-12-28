import torch

from typing import List, Tuple, Dict
from collections import defaultdict

from tkge.common.config import Config
from tkge.common.configurable import Configurable
from tkge.common.error import ConfigurationError
from tkge.data.dataset import DatasetProcessor

import enum


class Evaluation(Configurable):
    SPOT = enum.Enum('spot', ('s', 'p', 'o', 't'))

    def __init__(self, config: Config, dataset: DatasetProcessor):
        super().__init__(config=config)

        self.dataset = dataset

        self.device = self.config.get("task.device")
        self.filter = self.config.get("eval.filter")
        self.ordering = self.config.get("eval.ordering")
        self.k = self.config.get("eval.k")

        self.filtered_data = defaultdict(None)
        self.filtered_data['sp_'] = self.dataset.filter(type=self.filter, target='o')
        self.filtered_data['_po'] = self.dataset.filter(type=self.filter, target='s')

    def eval(self, queries: torch.Tensor, scores: torch.Tensor, miss='o'):
        metrics = {}

        filtered_list = self.filtered_data['sp_'] if miss == 'o' else self.filtered_data['_po']
        filtered_index = self.filter_query(queries, filtered_list)
        targets = queries[:, 2].long() if miss == 'o' else queries[:, 0].long()

        ranks = self.ranking(scores, targets, filtered_index)

        metrics['mean_ranking'] = self.mean_ranking(ranks)
        metrics['mean_reciprocal_ranking'] = self.mean_reciprocal_ranking(ranks)

        hits = self.hits(ranks)
        for k, hit_at_k in zip(self.k, hits):
            metrics[f"hits_at_{k}"] = hit_at_k

        return metrics

    def ranking(self, scores: torch.Tensor, targets: torch.Tensor, filtered_index: torch.Tensor):
        query_size = scores.size(0)
        vocabulary_size = scores.size(1)

        target_scores = scores[range(query_size), targets].unsqueeze(1).repeat((1, vocabulary_size))

        scores[filtered_index[0], filtered_index[1]] = 0.0

        if self.ordering == "optimistic":
            comp = scores.gt(target_scores)
        else:
            comp = scores.ge(target_scores)

        ranks = comp.sum(1) + 1

        return ranks.float()

    def filter_query(self, queries: torch.Tensor, filtered_list: Dict[str, List], miss: str = "o"):
        filtered_index = [[], []]
        for i, q in enumerate(queries):
            # TODO(gengyuan) formatting
            sid = int(q[0])
            rid = int(q[1])
            oid = int(q[2])

            if miss == "o":
                query = f"{sid}-{rid}-None"
            else:
                query = f"None-{rid}-{oid}"

            for j in filtered_list[query]:
                filtered_index[0].append(i)
                filtered_index[1].append(j)

        filtered_index = torch.Tensor(filtered_index).long().to(self.device)

        return filtered_index

    def mean_ranking(self, ranks):
        mr = torch.mean(ranks).item()

        return mr

    def mean_reciprocal_ranking(self, ranks):
        mrr = torch.mean(1. / ranks).item()

        return mrr

    def hits(self, ranks):
        hits_at = list(map(lambda x: torch.mean((ranks <= x).float()).item(), self.k))

        return hits_at
