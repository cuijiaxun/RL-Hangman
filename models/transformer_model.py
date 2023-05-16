import os
import sys

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote
import rlmeta.utils.nested_utils as nested_utils
from rlmeta.agents.ppo.ppo_model import PPOModel
from rlmeta.core.model import DownstreamModel, RemotableModel
from rlmeta.core.server import Server

class PPOTransformerModel(PPOModel):
    def __init__(self,
                 letter_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.letter_dim = letter_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        self.step_embed_dim = step_embed_dim
        self.input_dim = (self.letter_dim * 30 +
                          self.action_embed_dim + self.step_embed_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)
        self.position_embed = nn.Embedding(31,
                                           4)
        self.life_embed = nn.Embedding(7, 4)

        self.linear_i = nn.Linear(61, self.hidden_dim) #nn.Linear(self.input_dim, self.hidden_dim)
        # self.linear_o = nn.Linear(self.hidden_dim * self.window_size,
        #                           self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                   nhead=8,
                                                   dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.linear_a = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def make_one_hot(self, src: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = F.one_hot(src, num_classes)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def make_embedding(self, src: torch.Tensor,
                       embed: nn.Embedding) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = embed(src)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #obs = obs.to(torch.int64)
        assert obs.dim() == 3

        # batch_size = obs.size(0)
        #unbind_obs = torch.unbind(obs, dim=-1)

        #ls = unbind_obs[:-2]
        #act = unbind_obs[-2]
        #stp = unbind_obs[-1]

        #act = self.make_embedding(act, self.action_embed)
        #stp = self.make_embedding(stp, self.step_embed)
        
        #x = torch.cat((act, stp), dim=-1)
        #print(ls) 
        #for l in ls[::-1]:
        #    l_transform = self.make_one_hot(l, self.letter_dim)
        #    x = torch.cat((l_transform,x), dim=-1)
        #letter = obs[:,:,:27]
        #position = self.position_embed(obs[:,:,27].to(torch.int64))
        #guessed = obs[:,:,28:54]
        #max_life = self.life_embed(obs[:,:,54].to(torch.int64))
        letter, position, guessed, max_life = torch.tensor_split(obs,(27,28,54), dim=-1)
        invalid_action = torch.unbind(guessed, dim =-2)[0]
        #print("1:",torch.unbind(guessed, dim=-2)[0])
        #print("2:",torch.unbind(guessed, dim=-2)[1])
        assert letter.size(2) == 27
        assert guessed.size(2) == 26
        position = self.position_embed(position.to(torch.int64).squeeze(2))
        assert position.dim()==3
        max_life = self.life_embed(max_life.to(torch.int64).squeeze(2))
        assert max_life.dim()==3
        x = torch.cat((letter, guessed, position, max_life), dim=-1)
        #x = obs
        x = self.linear_i(x)
        x = x.transpose(0, 1).contiguous()
        h = self.encoder(x)
        # h = self.linear_o(h.view(batch_size, -1))
        h = h.mean(dim=0)

        p = self.linear_a(h)
        if invalid_action is not None:
            invalid_action = invalid_action.to(p.get_device())
            min_value = p.max() - p.min() + 1.0
            p = p - invalid_action * min_value

        logpi = F.log_softmax(p, dim=-1)
        v = self.linear_v(h)

        return logpi, v

    @remote.remote_method(batch_size=128)
    def act(
            self, obs: torch.Tensor, deterministic_policy: torch.Tensor, invalid_action: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            x = obs.to(self._device)
            d = deterministic_policy.to(self._device)
            logpi, v = self.forward(x)
            
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), v.cpu()


