from torch import nn
import torch

from .ddpg_modeling import Actor, Critic


class LunarActor(Actor,nn.Module):
    r"""
    Network to be used as actor in a DDPG algorithm.
    """
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dimension, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, action_dimension),
                                 nn.Tanh()
                                 )

    def forward(self, states: torch.Tensor):
        return self.net(states)


class LunarCritic(Critic,nn.Module):
    r"""
    Network to be used as critic in a DDPG algorithm.
    """
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        self.l1 = nn.Linear(state_dimension, 400)
        self.l2 = nn.Linear(400 + action_dimension, 200)
        self.l3 = nn.Linear(200, 1)

        self.relu = nn.ReLU()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.l1(states)
        x = self.relu(x)
        x = self.l2(torch.cat([x, actions], dim=1))
        x = self.relu(x)
        x = self.l3(x)
        return x

