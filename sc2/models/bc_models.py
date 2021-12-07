#!/usr/bin/env python
# Created at 2020/2/15
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from torch.nn import functional as F

def resolve_activate_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "sigmoid":
        return nn.Sigmoid
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    if name.lower() == "prelu":
        return nn.PReLU
    if name.lower() == "softmax":
        return nn.Softmax
    if name.lower() == "tanh":
        return nn.Tanh

class BCConfig:
    """ base BC config"""

    def __init__(self, num_states, num_actions, num_discrete_actions=0, discrete_actions_sections: Tuple = (0,), num_hiddens: Tuple = (64, 64), activation: str = "relu", **kwargs):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_discrete_actions = num_discrete_actions
        self.discrete_actions_sections = discrete_actions_sections
        self.num_hiddens = num_hiddens
        self.activation = activation
        for k, v in kwargs.items():
            setattr(self, k, v)


class BC(nn.Module):
    def __init__(self, config):

        super(BC, self).__init__()
        self.config = config
        # set up state space and action space
        self.num_states = config.num_states
        self.num_actions = config.num_actions
        self.num_hiddens = config.num_hiddens
        # set up discrete action info
        self.num_discrete_actions = config.num_discrete_actions
        assert sum(config.discrete_actions_sections) == self.num_discrete_actions, f"Expected sum of discrete actions's " \
                                                                       f"dimension =  {self.num_discrete_actions}"
        self.discrete_action_sections = config.discrete_actions_sections

        # set up module units
        _module_units = [self.num_states]
        _module_units.extend(self.num_hiddens)
        _module_units += self.num_actions,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(config.activation)

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            n_units_in, n_units_out = module_unit
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
        if self.num_discrete_actions:
            self._module_list.add_module(f"Layer_{idx + 1}_Custom_Softmax",
                                         MultiSoftMax(0, self.num_discrete_actions, self.discrete_action_sections))

    def get_block_size(self):
        return 1 # linghui: for vanilla bc, there is no need to make the context length

    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        """
        give states, calculate the distribution of actions
        :param x: unsqueezed states
        :return: xxx
        """
        x = states
        for module in self._module_list:
            x = module(x)
        logits = x
        loss = None
        if targets is not  None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), torch.tensor(targets, dtype=torch.long).reshape(-1)) # maybe we can multiply the rtgs on the loss
        return logits, loss

class MultiSoftMax(nn.Module):
    r"""customized module to deal with multiple softmax case.
    softmax feature: [dim_begin, dim_end)
    sections define sizes of each softmax case

    Examples::

        >>> m = MultiSoftMax(dim_begin=0, dim_end=5, sections=(2, 3))
        >>> input = torch.randn((2, 5))
        >>> output = m(input)
    """

    def __init__(self, dim_begin: int, dim_end: int, sections: Tuple = None):
        super().__init__()
        self.dim_begin = dim_begin
        self.dim_end = dim_end
        self.sections = sections

        if sections:
            assert dim_end - dim_begin == sum(sections), "expected same length of sections and customized" \
                                                         "dims"

    def forward(self, input_tensor: torch.Tensor):
        x = input_tensor[..., self.dim_begin:self.dim_end]
        res = input_tensor.clone()
        res[..., self.dim_begin:self.dim_end] = torch.cat([
            xx.softmax(dim=-1) for xx in torch.split(x, self.sections, dim=-1)], dim=-1)
        return res