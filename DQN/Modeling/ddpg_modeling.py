import random
from abc import abstractmethod, ABC
from typing import Any, Type, List, Tuple

import gym
import numpy as np

from .dqn_modeling import AgentEpisodicDQNTrainer, AgentDQN
from .trainer_modeling import AgentEpisodicTrainer, Agent, GridSearcher

from torch import nn
import torch


class Actor(ABC, nn.Module):
    r"""
    Base class for actor modules.
    """

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        pass


class Critic(ABC, nn.Module):
    r"""
    Base class for critic modules.
    """

    @abstractmethod
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        r"""
        Receive as input the states and actions to critic

        Args:
            states(torch.Tensor) : States. Tensor with dimension (batch_size,state_dimension)
            actions(torch.Tensor) : Actions. Tensor with dimension (batch_size, action_dimension)

        Returns:
            outputs(torch.Tensor) : Q values. Tensor with dimension (batch_size,1)
        """
        pass


class NoiseGenerator(ABC):
    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class LowPassFilteredNoise(NoiseGenerator):
    def __init__(self,
                 dimension: int,
                 mu: float = 0.15,
                 sigma: float = 0.2):
        assert 1 > mu >= 0
        self.dimension = dimension
        self.mu = mu
        self.sigma = sigma
        self.previous_val = torch.zeros(self.dimension)

    def generate(self) -> torch.Tensor:
        means = torch.zeros(self.dimension)
        std = torch.ones(self.dimension) * self.sigma

        to_return = - self.mu * self.previous_val + torch.normal(mean=means, std=std)
        self.previous_val = to_return

        return to_return

    def reset(self):
        self.previous_val = torch.zeros(self.dimension)


class AgentDDPG(Agent, nn.Module):
    def __init__(self,
                 critic_network: Type[Critic],
                 actor_network: Type[Actor],
                 network_device: torch.device = None,
                 critic_network_initialization_parameters: dict = None,
                 actor_network_initialization_parameters: dict = None,
                 noise_generator: NoiseGenerator = None
                 ):
        super().__init__()

        self.noise_generator = noise_generator
        # Set to nothing in case no parameters are necessary
        if critic_network_initialization_parameters is None:
            critic_network_initialization_parameters = {}
        if actor_network_initialization_parameters is None:
            actor_network_initialization_parameters = {}

        # Initialize the networks
        self.online_critic = critic_network(**critic_network_initialization_parameters).to(network_device)
        self.online_actor = actor_network(**actor_network_initialization_parameters).to(network_device)

        self.target_critic = critic_network(**critic_network_initialization_parameters).to(network_device)
        self.target_actor = actor_network(**actor_network_initialization_parameters).to(network_device)

        # Set target and online to the same
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        self.target_actor.load_state_dict(self.online_actor.state_dict())

        # Actor receives as input a vector with dimension (state_dim)
        # Critic receives as input a vector with dimension (state_dim + action_dim)

    def forward(self, state: torch.Tensor, epsilon: float = 0, device: torch.device = "cpu") -> torch.Tensor:
        """
        Agent that takes a decision given an action with an epsilon greedy policy.

        Args:
            state (torch.Tensor): The state from which decision has to be taken.
            epsilon (float): The epsilon to use for the decision
        Return:
            action (torch.Tensor): The action to be taken. It is a vector
        """

        # Here we have a continious action space.
        # We will ignore epsilon and just add a noise if it is different than 0
        # Move all tensor to cpu and detach since we want to
        if epsilon != 0:
            to_return = self._act(state,device) + self.noise_generator.generate()
            return to_return.numpy()
        else:
            return self._act(state,device).numpy()

    def _act(self,state,device):
        s_t: torch.Tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        action = self.online_actor(s_t).cpu().detach()
        return action



    def backward(self, transitions: List[Tuple], discount_factor: float = 1,
                 device: torch.device = "cpu") -> torch.Tensor:
        r"""
        Compute the MSE for the critic
        """
        s_t, a_t, r_t, dones_t, s_t_next = self._transform_to_tensors(transitions, device)

        target_q_values = self.target_critic(s_t_next, self.target_actor(s_t_next))  # (batch_size,1)
        targets = r_t + discount_factor * (1 - dones_t) * target_q_values  # (batch_size,1)

        # Compute the loss of the critic
        q_values = self.online_critic(s_t, a_t)  # (batch_size,1)
        loss = nn.functional.mse_loss(q_values, targets)

        return loss

    def save(self, path, extra_data) -> None:
        pass

    def align_networks(self, tau: float) -> None:
        self.target_critic = self.soft_updates(self.online_critic, self.target_critic, tau)
        self.target_actor = self.soft_updates(self.online_actor, self.target_actor, tau)

    def soft_updates(self, network: nn.Module, target_network: nn.Module, tau: float) -> nn.Module:
        """ Performs a soft copy of the network's parameters to the target
        network's parameter

        Args:
            network (nn.Module): neural network from which we want to copy the
                parameters
            target_network (nn.Module): network that is being updated
            tau (float): time constant that defines the update speed in (0,1)

        Returns:
            target_network (nn.Module): the target network

        """
        tgt_state = target_network.state_dict()
        for k, v in network.state_dict().items():
            tgt_state[k] = (1 - tau) * tgt_state[k] + tau * v
        target_network.load_state_dict(tgt_state)
        return target_network

    def actor_loss(self, transitions: List[Tuple], device) -> torch.Tensor:
        s_t, a_t, r_t, dones_t, s_t_next = self._transform_to_tensors(transitions, device)

        q_val = self.online_critic(s_t, self.online_actor(s_t))  # (batch_size ,1)
        return -torch.sum(q_val) / q_val.size()[0]

    def _transform_to_tensors(self, transitions: List[Tuple], device) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Transform to list and then tensor
        obses = np.array([np.array(t[0]) for t in transitions])  # (batch_size x state_dimension)
        actions = np.asarray([t[1] for t in transitions])  # (batch_size x action_dimension)
        rewards = np.asarray([t[2] for t in transitions])  # (batch_size,)
        dones = np.asarray([t[3] for t in transitions])  # (batch_size,)
        new_obses = np.asarray([t[4] for t in transitions])  # (batch_size x state_dimension)
        # todo : move the to statements out of here to the update_agent of the trainer
        s_t = torch.as_tensor(obses, dtype=torch.float32).to(device)  # (batch_size, state_dim)
        a_t = torch.as_tensor(actions, dtype=torch.int64).to(device)  # (batch_size, action_dim)
        r_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size,1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch_size,1)
        s_t_next = torch.as_tensor(new_obses, dtype=torch.float32).to(device)  # (batch_size, state_dim)

        return s_t, a_t, r_t, dones_t, s_t_next

    def to(self, device):
        self.online_critic.to(device)
        self.online_actor.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)


class AgentEpisodicDDPGTrainer(AgentEpisodicTrainer):

    def __init__(self,
                 environment: gym.Env,
                 agent: AgentDDPG,
                 learning_rate_initial: float = 5e-4,
                 learning_rate_actor:float = 5e-4,
                 device: torch.device = None,
                 target_update_frequency: int = 200,
                 target_update_strategy: str = "step",
                 clip_gradients: bool = False,
                 clipping_value: float = 2,
                 tau : float = 1e-3,
                 *args,
                 **kwargs):
        super().__init__(environment=environment, agent=agent, *args, **kwargs)
        assert issubclass(agent.__class__,AgentDDPG)
        self.tau=tau

        self.target_update_strategy = target_update_strategy
        self.target_update_frequency = target_update_frequency

        self.optimizer = torch.optim.Adam(self.agent.online_critic.parameters(), lr=learning_rate_initial)
        self.actor_optimizer = torch.optim.Adam(self.agent.online_actor.parameters(),lr=learning_rate_actor)

        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.agent.online_critic.parameters(),clipping_value)
            torch.nn.utils.clip_grad_norm_(self.agent.online_actor.parameters(),clipping_value)

        # Move the agent to the GPU
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.agent.to(self.device)

    def update_agent(self):
        # Sample from the buffer replay and compute loss
        transitions = random.sample(self.replay_buffer, self.batch_size)
        critic_loss = self.agent.backward(transitions, self.discount_factor, self.device)

        # Take a step (update the critic) (online network)
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update network every update steps or episodes
        if self.step % self.target_update_frequency and self.target_update_strategy == "step":
            actor_loss = self.agent.actor_loss(transitions,self.device)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.agent.align_networks(self.tau)
        elif self.episode % self.target_update_frequency:
            actor_loss = self.agent.actor_loss(transitions, self.device)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.agent.align_networks(self.tau)

    def action_agent_callback(self, state: Any, epsilon: float):
        return self.agent.forward(state, epsilon, self.device)
