import torch
import torch.nn.functional as F

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.error import ConfigurationError

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar

T = TypeVar("T", bound="Loss")


class Loss(metaclass=ABC, Registrable, Configurable):
    def __init__(self, config: Config):
        Registrable.__init__()
        Configurable.__init__(config=config)

        self.config = config
        self._loss = None

    @classmethod
    def create(config: Config, purge: bool= True) -> T:
        """Factory method for loss"""

        loss_type = config.get("train.loss.type")
        load_default = config.get("train.loss.load_default")


        if loss_type in Loss.list_available():
            # kwargs = config.get("train.loss_arg")  # TODO: 需要改成key的格式
            purged_config = Loss.by_name(loss_type)._parse_config(config, load_default) if purge else config


            return Loss.by_name(loss_type)(config)
        else:
            raise ConfigurationError(
                f"{loss_type} specified in configuration file is not supported"
                f"implement your loss class with `Loss.register(name)"
            )


    @abstractmethod
    def __call__(self, scores, labels, **kwargs):
        """Computes the loss given the scores and corresponding labels.

        `scores` is a batch_size x triples matrix holding the scores predicted by some
        model.

        `labels` is either (i) a batch_size x triples Boolean matrix holding the
        corresponding labels or (ii) a vector of positions of the (then unique) 1-labels
        for each row of `scores`.

        """
        raise NotImplementedError()

    def _labels_as_matrix(self, scores, labels):
        """Reshapes `labels` into indexes if necessary.

        See `__call__`. This function converts case (ii) into case (i).
        """
        if labels.dim() == 2:
            return labels
        else:
            x = torch.zeros(
                scores.shape, device=self.config.get("task.device"), dtype=torch.float
            )
            x[range(len(scores)), labels] = 1.0
            return x

    def _labels_as_indexes(self, scores, labels):
        """Reshapes `labels` into matrix form if necessary and possible.

        See `__call__`. This function converts case (i) into case (ii). Throws an error
        if there is a row which does not have exactly one 1.

        """
        if labels.dim() == 1:
            return labels
        else:
            x = labels.nonzero()
            if not x[:, 0].equal(
                    torch.arange(len(labels), device=self.config.get("job.device"))
            ):
                raise ValueError("exactly one 1 per row required")
            return x[:, 1]
