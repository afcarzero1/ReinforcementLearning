import os
from datetime import datetime
from typing import Any, List, Tuple, Optional, Union

import gym
import numpy as np
import random
from sklearn.model_selection import ParameterGrid
from termcolor import colored

from trainer_modeling import AgentEpisodicTrainer, Agent

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
        type_network = self.online_network.__class__
        if network_initialization_parameters is None:
            network_initialization_parameters = {}

        if network_device is None:
            network_device = "cuda" if torch.cuda.is_available() else "cpu"
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
        list_of_obs = [t[0] for t in transitions]

        # Transform to list and then tensor
        obses = np.array([np.array(t[0]) for t in transitions])  # (batch_size x state_dimension)
        actions = np.asarray([t[1] for t in transitions])  # (batch_size,)
        rewards = np.asarray([t[2] for t in transitions])  # (batch_size,)
        dones = np.asarray([t[3] for t in transitions])  # (batch_size,)
        new_obses = np.asarray([t[4] for t in transitions])  # (batch_size x state_dimension)
        # todo : move the to statements out of here
        obses_t = torch.as_tensor(obses, dtype=torch.float32).to(device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)  # (batch_size,1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size,1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size,1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).to(device)

        target_q_values = self.target_network(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + discount_factor * (1 - dones_t) * max_target_q_values

        # Compute the loss
        q_values = self.online_network(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

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

    def align_networks(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def to(self, device):
        super(AgentDQN, self).to(device)
        self.online_network = self.online_network.to(device)
        self.target_network = self.target_network.to(device)

    def save(self, path, extra_data):
        torch.save(self.online_network.state_dict(), path)

    def load(self,path):
        self.online_network=torch.load(path)


class Network(nn.Module):
    def __init__(self, env):
        super(Network, self).__init__()
        in_features = int(np.prod(env.observation_space.shape))  # Number of inputs in the layer

        self.net = nn.Sequential(nn.Linear(in_features, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t: torch.Tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self.forward(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action


class AgentEpisodicDQNTrainer(AgentEpisodicTrainer):
    def __init__(self,
                 environment: gym.Env,
                 agent: AgentDQN,
                 learning_rate_initial: float = 5e-4,
                 device: torch.device = None,
                 target_update_frequency: int = 200,
                 *args,
                 **kwargs):
        super(AgentEpisodicDQNTrainer, self).__init__(environment=environment, agent=agent, *args, **kwargs)

        # Set the frequency with which update the target.
        self.target_update_frequency = target_update_frequency

        assert type(self.agent) == AgentDQN
        self.optimizer = torch.optim.Adam(self.agent.online_network.parameters(), lr=learning_rate_initial)

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

        # Update network every update steps
        if self.step % self.target_update_frequency:
            self.agent.align_networks()

    def action_agent_callback(self, state: Any, epsilon: float):
        return self.agent.forward(state, epsilon, self.device)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="human")
    env = gym.make('LunarLander-v2')
    dim_state = len(env.observation_space.high)
    env.reset()

    q_learner = Network(env)

    agent = AgentDQN(network=q_learner,
                     env=env,
                     network_initialization_parameters={"env": env})

    trainer = AgentEpisodicDQNTrainer(env,
                                      agent,
                                      discount_factor=0.99,
                                      learning_rate_initial=5e-4,
                                      batch_size=64,
                                      buffer_size=500,
                                      buffer_size_min=64)

    trainer.train()
    trainer.test()
    trainer.agent.save("network.pth",extra_data=None)
    env = gym.make('LunarLander-v2', render_mode="human")
    trainer = AgentEpisodicDQNTrainer(env, agent)
    trainer.play_game()
