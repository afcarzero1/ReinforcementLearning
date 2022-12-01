import os
from datetime import datetime
from typing import Any, List, Tuple

import gym
import numpy as np
from sklearn.model_selection import ParameterGrid
from termcolor import colored

from trainer_modeling import AgentEpisodicTrainer, Agent

import torch
from torch import nn


class AgentDQN(Agent,nn.Module):
    def __init__(self, network: nn.Module, env: gym.Env, network_initialization_parameters: dict = None):
        super().__init__()
        self.online_network = network
        self.action_space = env.action_space
        # Get the class of the network
        # Create the parallel version and load the weights
        type_network = self.online_network.__class__
        if network_initialization_parameters is None:
            network_initialization_parameters = {}
        self.target_network = type_network(**network_initialization_parameters)

        self.target_network.load_state_dict(self.online_network.state_dict())

    def forward(self, state: Any, epsilon: float = 0) -> Any:
        if np.random.binomial(size=1, n=1, p=epsilon) == 1:
            # Take greedy action from the action space
            self.action_space.sample()
        else:
            # Take action
            return self._act(state)

    def backward(self, transitions: List[Tuple], discount_factor: float = 1, device: torch.device = "cpu"):
        # Generate the targets
        list_of_obs = [t[0] for t in transitions]

        # Transform to list and then tensor
        obses = np.array([np.array(t[0]) for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32).to(device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).to(device)

        target_q_values = self.target_network(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_t + discount_factor * (1 - dones_t) * max_target_q_values

        # Compute the loss
        q_values = self.online_network(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    def _act(self, state):
        """
        Act according to learnt behaviour of online network
        :param state:
        :return:
        """
        obs_t: torch.Tensor = torch.as_tensor(state, dtype=torch.float32)
        q_values = self.online_network.forward(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

    def align_networks(self):
        self.target_network.load_state_dict(self.online_network.state_dict())


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
                 learning_rate_initial: float = 0.001,
                 device : torch.device = None,
                 *args,
                 target_update_frequency: int = 20,
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
        transitions = np.random.sample(self.replay_buffer, self.batch_size)
        loss = self.agent.backward(transitions, self.discount_factor, self.device)

        # Take a step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update network every update steps
        if self.step % self.target_update_frequency:
            self.agent.align_networks()


class GridSearcher:
    def __init__(self,
                 env,
                 agent_class,
                 agent_trainer_class,
                 number_episodes=500):
        self.env = env
        self.agent_class = agent_class
        self.agent_trainer_class = agent_trainer_class
        self.number_episodes = number_episodes

        self.results = []

    def grid_search(self, agent_parameters: dict, trainer_parameters: dict = {}):
        ### SET GENERAL PARAMETERS
        NUMBER_EPISODES = self.number_episodes
        if not os.path.exists(os.path.join(".", "RESULTS")):
            os.makedirs(os.path.join(".", "RESULTS"))

        ### CREATE ENVIRONMENT ###
        env = gym.make('MountainCar-v0')
        env.reset()

        max_reward = float("-inf")
        best_hyperparameters = {}
        stored_time = ""
        i = 0

        # Clean the results
        self.results = []

        for hyperparameters in ParameterGrid(agent_parameters):
            for trainer_hyp in ParameterGrid(trainer_parameters):
                i += 1
                print("{:5} / {:5} parameter".format(i, len(ParameterGrid(trainer_parameters)) * len(
                    ParameterGrid(agent_parameters))))

                trainer = self.train_step(hyperparameters, trainer_hyp, verbose=False)
                passed, avg_rew, conf = trainer.test()
                avg_rew_lim = avg_rew - conf

                self.results.append((hyperparameters, trainer_hyp, passed, avg_rew, conf))

                ### IF PASSED THE TEST SAVE THE MODEL ###
                if passed:
                    e = datetime.now()
                    time = e.strftime("%Y-%m-%d%H-%M-%S")

                    trainer.agent.save(file_prefix=os.path.join("RESULTS", time), extra_data=hyperparameters)
                    if avg_rew_lim > max_reward:
                        print(colored("[NEW BEST] The new best hyperparameter combination is:"))
                        print(hyperparameters)
                        max_reward = avg_rew_lim
                        best_hyperparameters = hyperparameters
                        stored_time = time

        ### RETURN BEST FOUND POLICY
        print(colored("The best policy, stored at time " + stored_time + " is:", 'red'))
        print(best_hyperparameters)
        return best_hyperparameters

    def train_step(self, model_parameters: dict, trainer_parameters={}, verbose=False):
        agent = self.agent_class(state_dimension=2,
                                 number_actions=self.env.action_space.n,
                                 **model_parameters)
        ### CREATE TRAINER ###
        trainer = self.agent_trainer_class(environment=self.env,
                                           agent=agent,
                                           number_episodes=self.number_episodes,
                                           episode_reward_trigger=-135,
                                           epsilon_initial=0.8,
                                           early_stopping=False,
                                           information_episodes=1000,
                                           **trainer_parameters)

        ### TRAIN AND TEST ###
        trainer.train(verbose=verbose)

        return trainer


if __name__ == '__main__':


    env = gym.make('LunarLander-v2')
    dim_state = len(env.observation_space.high)
    env.reset()

    q_learner = Network(env)

    agent = AgentDQN(network=q_learner,
                     env=env)

    trainer = AgentEpisodicDQNTrainer(env,agent)


    trainer.train()


