import os
import pickle
from datetime import datetime
from typing import Any, List, Tuple, Optional, Union

import gym
import numpy as np
import random

from matplotlib import pyplot as plt, cm
from sklearn.model_selection import ParameterGrid
from termcolor import colored

from .trainer_modeling import AgentEpisodicTrainer, Agent, GridSearcher

import torch
from torch import nn


class AgentDQN(Agent, nn.Module):
    def __init__(self,
                 network: nn.Module,
                 env: gym.Env,
                 network_device: torch.device = None,
                 network_initialization_parameters: dict = None):
        super().__init__()
        self.online_network = network
        self.action_space = env.action_space
        # Get the class of the network
        # Create the parallel version and load the weights
        # todo : use the class as parameter instead
        type_network = self.online_network.__class__
        if network_initialization_parameters is None:
            network_initialization_parameters = {}

        if network_device is None:
            network_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.online_network = type_network(**network_initialization_parameters).to(device=network_device)
        self.target_network = type_network(**network_initialization_parameters).to(device=network_device)

        self.target_network.load_state_dict(self.online_network.state_dict())

    def forward(self, state: Any, epsilon: float = 0, device: torch.device = "cpu") -> Any:
        if np.random.binomial(size=1, n=1, p=epsilon) == 1:
            # Take greedy action from the action space
            return self.action_space.sample()
        else:
            # Take action
            return self._act(state, device)

    def backward(self, transitions: List[Tuple], discount_factor: float = 1, device: torch.device = "cpu"):
        # Generate the targets
        self.online_network = self.online_network.to(device)  # TODO fix this

        # Transform to list and then tensor
        obses = np.array([np.array(t[0]) for t in transitions])  # (batch_size x state_dimension)
        actions = np.asarray([t[1] for t in transitions])  # (batch_size,)
        rewards = np.asarray([t[2] for t in transitions])  # (batch_size,)
        dones = np.asarray([t[3] for t in transitions])  # (batch_size,)
        new_obses = np.asarray([t[4] for t in transitions])  # (batch_size x state_dimension)
        # todo : move the to statements out of here to the update_agent of the trainer
        obses_t = torch.as_tensor(obses, dtype=torch.float32).to(device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)  # (batch_size,1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size,1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size,1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).to(device)

        target_q_values = self.target_network(new_obses_t)  # (batch_size,action_dim)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # (batch_size,1)

        targets = rewards_t + discount_factor * (1 - dones_t) * max_target_q_values  # (batch_size,1)

        # Compute the loss
        q_values = self.online_network(obses_t)  # (batch_size,action_dim)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)  # (batch_size,1)
        loss = nn.functional.mse_loss(action_q_values, targets)

        return loss

    def _act(self, state, device):
        """
        Act according to learnt behaviour of online network
        :param state:
        :return:
        """
        obs_t: torch.Tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        q_values = self.online_network.forward(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action


    def _q_val(self,state,device):
        obs_t: torch.Tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        q_values = self.online_network.forward(obs_t.unsqueeze(0))
        return q_values

    def align_networks(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def to(self, device):
        super(AgentDQN, self).to(device)
        self.online_network = self.online_network.to(device)
        self.target_network = self.target_network.to(device)

    def save(self, path, extra_data):
        r"""
        Save the online network as a .pth file.

        Args:
            path (str) : The path to use for saving the network
            extra_data (Any) : Additional Data
        """
        torch.save(self.online_network.state_dict(), path+"network.pth")

        with open(path+"extra.pkl","wb") as f:
            pickle.dump(extra_data,f)

    def load(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def plot_q(self,file_name="",device="cpu"):
        NUMBER_POINTS = 100


        y = np.linspace(0, 1.5, NUMBER_POINTS)
        w = np.linspace(-np.pi, np.pi, NUMBER_POINTS)

        value = np.zeros((NUMBER_POINTS, NUMBER_POINTS))

        for i in range(NUMBER_POINTS):
            for j in range(NUMBER_POINTS):

                q_v = self._q_val((0,y[i],0,0,w[j],0,0,0),device)
                value[(i,j)] = torch.max(q_v).item()

        x, y = np.meshgrid(y, w)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Height')
        ax.set_ylabel('Angle')
        ax.set_zlabel('Value')

        fig.suptitle('Value Function', fontsize=20)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if file_name == "":
            plt.show()
            plt.close()
        else:
            plt.savefig(file_name)
            plt.close()

    def plot_a(self,file_name="",device ="cpu"):
        NUMBER_POINTS = 100

        y = np.linspace(0, 1.5, NUMBER_POINTS)
        w = np.linspace(-np.pi, np.pi, NUMBER_POINTS)

        value = np.zeros((NUMBER_POINTS, NUMBER_POINTS))

        for i in range(NUMBER_POINTS):
            for j in range(NUMBER_POINTS):
                q_v = self._q_val((0, y[i], 0, 0, w[j], 0, 0, 0), device)
                value[(i, j)] = torch.argmax(q_v).item()

        x, y = np.meshgrid(y, w)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Height')
        ax.set_ylabel('Angle')
        ax.set_zlabel('Action')

        fig.suptitle('Action Function', fontsize=20)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if file_name == "":
            plt.show()
            plt.close()
        else:
            plt.savefig(file_name)
            plt.close()



class NetworkDQN(nn.Module):
    def act(self, obs):
        obs_t: torch.Tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self.forward(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action


class SmallNetwork(NetworkDQN):
    def __init__(self, env):
        super(SmallNetwork, self).__init__()
        in_features = int(np.prod(env.observation_space.shape))  # Number of inputs in the layer

        self.net = nn.Sequential(nn.Linear(in_features, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)


class BigNetwork(NetworkDQN):
    def __init__(self, env):
        super(BigNetwork, self).__init__()
        in_features = int(np.prod(env.observation_space.shape))  # Number of inputs in the layer

        self.net = nn.Sequential(nn.Linear(in_features, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)


class AgentEpisodicDQNTrainer(AgentEpisodicTrainer):
    def __init__(self,
                 environment: gym.Env,
                 agent: AgentDQN,
                 learning_rate_initial: float = 5e-4,
                 device: torch.device = None,
                 target_update_frequency: int = 200,
                 target_update_strategy : str = "step",
                 clip_gradients: bool = False,
                 clipping_value: float = 2,
                 *args,
                 **kwargs):
        super(AgentEpisodicDQNTrainer, self).__init__(environment=environment, agent=agent, *args, **kwargs)

        # Set the frequency with which update the target.
        self.target_update_strategy = target_update_strategy
        self.target_update_frequency = target_update_frequency

        assert type(self.agent) == AgentDQN
        self.optimizer = torch.optim.Adam(self.agent.online_network.parameters(), lr=learning_rate_initial)

        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.agent.online_network.parameters(), clipping_value)

        # Move the agent to the GPU
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.agent.to(device)

    def update_agent(self):
        # Sample from the buffer replay and compute loss
        transitions = random.sample(self.replay_buffer, self.batch_size)
        loss = self.agent.backward(transitions, self.discount_factor, self.device)

        # Take a step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update network every update steps or episodes
        if self.step % self.target_update_frequency and self.target_update_strategy=="step":
            self.agent.align_networks()
        elif self.episode % self.target_update_frequency:
            self.agent.align_networks()

    def action_agent_callback(self, state: Any, epsilon: float):
        return self.agent.forward(state, epsilon, self.device)


if __name__ == '__main__':
    pass