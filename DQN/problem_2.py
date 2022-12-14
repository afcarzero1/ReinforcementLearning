import gym
import numpy as np
import torch
from torch import nn

from Modeling.trainer_modeling import GridSearcher, BigHyperParameter
from Modeling.ddpg_modeling import AgentDDPG, Actor, Critic, AgentEpisodicDDPGTrainer, LowPassFilteredNoise


class LunarActor(Actor):
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


class LunarCritic(Critic):
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


def solve_problem():
    env = gym.make('LunarLander-v2', continuous=True)
    env.reset()

    agent = AgentDDPG(critic_network=LunarCritic,
                      actor_network=LunarActor,
                      critic_network_initialization_parameters={"state_dimension": np.prod(env.observation_space.shape),
                                                                "action_dimension": np.prod(env.action_space.shape)},
                      actor_network_initialization_parameters={"state_dimension": np.prod(env.observation_space.shape),
                                                               "action_dimension": np.prod(env.action_space.shape)}
                      , noise_generator=LowPassFilteredNoise(np.prod(env.action_space.shape),
                                                             sigma=2.5)
                      )

    trainer = AgentEpisodicDDPGTrainer(env,
                                       agent,
                                       number_episodes=500,
                                       discount_factor=0.99,
                                       learning_rate_initial=5e-4,
                                       learning_rate_actor=5e-5,
                                       batch_size=64,
                                       buffer_size=30000,
                                       buffer_size_min=30000,
                                       early_stopping=True,
                                       early_stopping_trigger=200,
                                       early_stopping_episodes_trigger=50,
                                       target_update_frequency=4,
                                       clipping_value=1,
                                       clip_gradients=True,
                                       tau=1e-3
                                       )

    trainer.train()
    trainer.plot_rewards()
    trainer.test(verbose=True, N=100)
    trainer.agent.save("ddpgnetwork", extra_data=None)
    env = gym.make('LunarLander-v2', continuous=True, render_mode="human")
    trainer = AgentEpisodicDDPGTrainer(env, agent)
    trainer.play_game()


def test_network(path_actor,path_critic):
    env = gym.make('LunarLander-v2', continuous=True)
    env.reset()

    agent = AgentDDPG(critic_network=LunarCritic,
                      actor_network=LunarActor,
                      critic_network_initialization_parameters={"state_dimension": np.prod(env.observation_space.shape),
                                                                "action_dimension": np.prod(env.action_space.shape)},
                      actor_network_initialization_parameters={"state_dimension": np.prod(env.observation_space.shape),
                                                               "action_dimension": np.prod(env.action_space.shape)}
                      , noise_generator=LowPassFilteredNoise(np.prod(env.action_space.shape),
                                                             sigma=2.5)
                      )

    agent.load(path_actor,path_critic)

    agent.to("cuda")
    agent.plot_pi(device="cuda")
    agent.plot_q(device="cuda")

    trainer = AgentEpisodicDDPGTrainer(env,
                                       agent)

    trainer.test(verbose=True,N=200)
    env = gym.make('LunarLander-v2', continuous=True,render_mode="human")
    trainer = AgentEpisodicDDPGTrainer(env, agent)
    trainer.play_game()


def find_best_hyperparameters():
    """
    Find the best hyper-parameters for the LunarLander gym environment using a DDPG algorithm.
    """
    env = gym.make('LunarLander-v2', continuous=True)
    env.reset()

    NUMBER_EPISODES = 500

    grid_searcher = GridSearcher(env, agent_class=AgentDDPG, agent_trainer_class=AgentEpisodicDDPGTrainer,
                                 save_all=True, folder="RESULTS_DDPG")

    lunar_critic_param = BigHyperParameter(LunarCritic, "LunarCritic")
    lunar_actor_param = BigHyperParameter(LunarActor, "LunarActor")

    agent_parameters = {"critic_network": [lunar_critic_param],
                        "actor_network": [lunar_actor_param],
                        "critic_network_initialization_parameters": [
                            {"state_dimension": np.prod(env.observation_space.shape),
                             "action_dimension": np.prod(env.action_space.shape)}],
                        "actor_network_initialization_parameters": [
                            {"state_dimension": np.prod(env.observation_space.shape),
                             "action_dimension": np.prod(env.action_space.shape)}]
        , "noise_generator": [LowPassFilteredNoise(np.prod(env.action_space.shape),
                                                   sigma=1,
                                                   decreasing_sigma=True,
                                                   final_sigma=0.1,
                                                   episodes=NUMBER_EPISODES),
                              LowPassFilteredNoise(np.prod(env.action_space.shape),
                                                   sigma=2),
                              LowPassFilteredNoise(np.prod(env.action_space.shape),
                                                   sigma=0.2)
                              ]

                        }

    trainer_parameters = {"discount_factor": [0.99],
                          "batch_size": [64],
                          "buffer_size": [30000, 100000],
                          "buffer_size_min": [30000],
                          "early_stopping": [True],
                          "target_update_frequency": [4, 2],
                          "target_update_strategy": ["step"],
                          "early_stopping_trigger": [200],
                          "early_stopping_episodes_trigger": [50],
                          "environment": [env],
                          "clip_gradients": [True],
                          "clipping_value": [1],
                          "tau": [1e-3],
                          "learning_rate_actor": [5e-5],
                          "learning_rate_initial": [5e-4],
                          "number_episodes": [NUMBER_EPISODES],
                          "reduce_noise": [True, False],
                          "reduce_noise_trigger": [150],
                          "reduce_noise_episodes_trigger": [30]
                          }

    grid_searcher.grid_search(agent_parameters, trainer_parameters)


def analyze_best_network(directory):
    best_agent_param,best_trainer_param = GridSearcher.analyze_results(directory)
    env = gym.make('LunarLander-v2', continuous=True)
    env.reset()
    grid_searcher = GridSearcher(env, agent_class=AgentDDPG, agent_trainer_class=AgentEpisodicDDPGTrainer,
                                 save_all=True, folder="RESULTS_DDPG")

    agent_param = {}
    for k, v in best_agent_param.items():
        if k == "actor_network":
            agent_param[k] = LunarActor
        elif k == "critic_network":
            agent_param[k] = LunarCritic
        else:
            agent_param[k] = v

    # Repeat the training for observing episode steps
    grid_searcher.train_step(agent_param,best_trainer_param,verbose=True)


    # See effect of discount factor
    agent_param = {k : [v] for k,v in agent_param.items()}
    trainer_param = {k : [v,1,0.1] if k == "discount_factor" else [v] for k,v in best_trainer_param.items()}
    grid_searcher = GridSearcher(env, agent_class=AgentDDPG, agent_trainer_class=AgentEpisodicDDPGTrainer,
                                 save_all=True, folder="RESULTS_DDPG_DISCOUNT")

    grid_searcher.grid_search(agent_param,trainer_param)


    # See effect of buffer size
    trainer_param = {k: [v, 500, 5000, 100000] if k == "buffer_size" else [v] for k, v in best_trainer_param.items()}
    grid_searcher = GridSearcher(env, agent_class=AgentDDPG, agent_trainer_class=AgentEpisodicDDPGTrainer,
                                 save_all=True, folder="RESULTS_DDPG_BUFFER")
    grid_searcher.grid_search(agent_param,trainer_param)





if __name__ == '__main__':

    #best_trainer = GridSearcher.analyze_results("RESULTS_DDPG/2022-12-1319-33-54/")
    #analyze_best_network("RESULTS_DDPG/2022-12-1319-33-54/")
    test_network("RESULTS_DDPG/2022-12-1319-33-54/009network_actor.pth","RESULTS_DDPG/2022-12-1319-33-54/009network_critic.pth")
    #test_network("ddpgnetwork.pthnetwork_actor.pth","ddpgnetwork.pthnetwork_critic.pth")
    #find_best_hyperparameters()
    #solve_problem()
