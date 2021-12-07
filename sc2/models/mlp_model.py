"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import copy
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    def __init__(self, state_size, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_size = state_size
        for k, v in kwargs.items():
            setattr(self, k, v)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, model_type='actor'):
        super().__init__()

        self.config = config

        self.block_size = config.block_size
        self.model_type = config.model_type
        self.state_size = config.state_size

        self._layer_N = 2
        self.feature_norm = nn.LayerNorm(self.state_size)

        active_func = nn.ReLU()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain('relu')

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(self.state_size, 64)), active_func, nn.LayerNorm(64))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(64, 64)), active_func, nn.LayerNorm(64))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

        if model_type == 'actor':
            self.head = nn.Linear(64, config.vocab_size, bias=False)
        elif model_type == 'critic':
            self.head = nn.Linear(64, 1, bias=False)
        else:
            raise NotImplementedError

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self, train_config, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    # state, action, and return
    def forward(self, states, pre_actions, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, block_size, 1)

        x = states.view(-1, self.state_size)
        x = self.feature_norm(x)
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x = self.head(x)
        logits = x.view(-1, 1, x.size(-1))

        return logits
