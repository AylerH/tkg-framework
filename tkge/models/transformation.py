from torch import nn
import torch

from abc import ABC, abstractmethod
from typing import Dict

from tkge.common.registrable import Registrable
from tkge.common.configurable import Configurable
from tkge.common.config import Config
from tkge.common.paramtype import *
from tkge.common.error import ConfigurationError


class Transformation(ABC, nn.Module, Registrable, Configurable):
    def __init__(self, config: Config):
        nn.Module.__init__(self)
        Registrable.__init__(self)
        Configurable.__init__(self, config=config)

    @classmethod
    def create_from_name(cls, config: Config):
        transformation_type = config.get("model.transformation.type")
        kwargs = config.get("model.transformation.args")

        kwargs = kwargs if not isinstance(kwargs, type(None)) else {}

        if transformation_type in Transformation.list_available():
            # kwargs = config.get("model.args")  # TODO: get all args params
            return Transformation.by_name(transformation_type)(config, **kwargs)
        else:
            raise ConfigurationError(
                f"{transformation_type} specified in configuration file is not supported "
                f"implement your model class with `Transformation.register(name)"
            )

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError


    @staticmethod
    @abstractmethod
    def embedding_constraint():
        raise NotImplementedError


@Transformation.register(name="translation_tf")
class TranslationTransformation(Transformation):
    gamma = NumberParam('gamma', default_value=100)
    p = NumberParam('p', default_value=1)

    def __init__(self, config):
        super(TranslationTransformation, self).__init__(config=config)

        self.p = self.config.get('model.transformation.p')
        self.gamma = self.config.get('model.transformation.gamma')

    def forward(self, head, rel, tail):
        scores = head['real'] + rel['real'] - tail['real']
        scores = self.gamma - torch.norm(scores, p=self.p, dim=1)

        return scores

    @staticmethod
    def embedding_constraint():
        """
        Translation only support real embeddings.
        """
        constraints = {'entity': ['real'],
                       'relation': ['real']}

        return constraints


@Transformation.register(name="rotation_tf")
class RotationTransformation(Transformation):
    gamma = NumberParam('gamma', default_value=100)

    def __init__(self, config):
        super(RotationTransformation, self).__init__(config=config)

        self.gamma = self.config.get('gamma')
        self.range = self.config.get('range')

    def forward(self, head: Dict[str, torch.Tensor], rel: Dict[str, torch.Tensor], tail: Dict[str, torch.Tensor]):
        """
        head and tail should be Dict[str, torch.Tensor]
        """
        pi = 3.14159265358979323846

        phase_rel = rel['real'] / self.range * pi

        real_rel = torch.cos(phase_rel)
        imag_rel = torch.sin(phase_rel)

        re_score = real_rel * tail['real'] + imag_rel * tail['imag']
        im_score = real_rel * tail['imag'] - imag_rel * tail['real']
        re_score = re_score - head['real']
        im_score = im_score - head['imag']

        scores = torch.stack([re_score, im_score], dim=0)
        scores = scores.norm(dim=0)

        scores = self.gamma - scores.sum(dim=-1)
        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real', 'imag'],
                       'relation': ['real', 'imag']}


@Transformation.register(name="rigid_tf")
class RigidTransformation(Transformation):
    pass


@Transformation.register(name="bilinear_tf")
class BilinearTransformation(Transformation):
    pass


@Transformation.register(name="distmult_tf")
class DistMult(Transformation):
    dropout = NumberParam('dropout', default_value=0.4)  # range (0, 1)

    def __init__(self, config):
        super(DistMult, self).__init__(config=config)

        self.dropout = config.get('dropout')

    def forward(self, head, rel, tail, summation='True'):
        scores = head * rel * tail

        if summation:
            scores = scores.sum(dim=-1)

        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real'],
                       'relation': ['real']}

        return constraints


@Transformation.register(name="complex_factorization_tf")
class ComplexFactorizationTransformation(Transformation):
    def __init__(self, config):
        super(ComplexFactorizationTransformation, self).__init__(config=config)

        self.flatten = config.get('flatten')

    def _forward(self, input: Dict):
        assert 'head' in input
        assert 'rel' in input
        assert 'tail' in input

    def forward(self, U: Dict, V: Dict, W: Dict):
        """
        U, V, W should be Dict[str, torch.Tensor], keys are 'real' and 'imag'
        """
        assert isinstance(U, dict)
        assert isinstance(V, dict)
        assert isinstance(W, dict)

        assert ['real', 'imag'] in U.keys()
        assert ['real', 'imag'] in V.keys()
        assert ['real', 'imag'] in W.keys()

        if self.flatten:
            scores = (U['real'] * V['real'] - U['imag'] * V['imag']) * W['real'].t() + \
                     (U['imag'] * V['real'] + U['real'] * V['imag']) * W['imag'].t()

        else:
            scores = (U['real'] * V['real'] - U['imag'] * V['imag']) @ W['real'].t() + \
                     (U['imag'] * V['real'] + U['real'] * V['imag']) @ W['imag'].t()

        return scores

    @staticmethod
    def embedding_constraint():
        constraints = {'entity': ['real', 'imag'],
                       'relation': ['real', 'imag']}

        return constraints
