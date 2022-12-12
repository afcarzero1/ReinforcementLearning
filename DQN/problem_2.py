import gym
import numpy as np
import torch
from torch import nn

from Modeling.ddpg_modeling import AgentDDPG, Actor, Critic, AgentEpisodicDDPGTrainer, LowPassFilteredNoise


class LunarActor(Actor):
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


class LunarCritic(Critic):
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        self.l1 = nn.Linear(state_dimension, 400)
        self.l2 = nn.Linear(400 + action_dimension, 200)
        self.l3 = nn.Linear(200, 1)

        self.relu = nn.ReLU()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.l1(states)
        x = self.relu(x)
        x = self.l2(torch.cat([x, actions], dim=0))
        x = self.relu(x)
        x = self.l3(x)
        return x


def solve_problem():
    env = gym.make('LunarLander-v2', continious=True)
    env.reset()

    agent = AgentDDPG(critic_network=LunarCritic,
                      actor_network=LunarActor,
                      critic_network_initialization_parameters={"state_dimension": np.prod(env.observation_space.shape),
                                                                "action_dimension": np.prod(env.action_space.shape)},
                      actor_network_initialization_parameters={"state_dimension": np.prod(env.observation_space.shape),
                                                               "action_dimension": np.prod(env.action_space.shape)}
                      , noise_generator= LowPassFilteredNoise(np.prod(env.action_space.shape)[0])
                      )

    trainer = AgentEpisodicDDPGTrainer(env,
                                       agent,
                                       discount_factor=0.99,
                                       learning_rate_initial=5e-4,
                                       learning_rate_actor=5e-5,
                                       batch_size=64,
                                       buffer_size=30000,
                                       buffer_size_min=30000,
                                       early_stopping=True,
                                       early_stopping_trigger=200,
                                       early_stopping_episodes_trigger=50,
                                       target_update_frequency=2,
                                       clipping_value=1,
                                       clip_gradients=True,
                                       )

    trainer.train()
    trainer.plot_rewards()
    trainer.test(verbose=True, N=100)
    trainer.agent.save("network.pth", extra_data=None)
    env = gym.make('LunarLander-v2', render_mode="human")
    trainer = AgentEpisodicDDPGTrainer(env, agent)
    trainer.play_game()
