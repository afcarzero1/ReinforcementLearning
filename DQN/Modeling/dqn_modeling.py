import os
from datetime import datetime

import gym
import numpy as np
from sklearn.model_selection import ParameterGrid
from termcolor import colored

from trainer_modeling import AgentEpisodicTrainer

import torch

class AgentEpisodicDQNTrainer(AgentEpisodicTrainer):


    def update_agent(self):
        transitions = np.random.sample(self.replay_buffer, self.batch_size)

        list_of_obs = [t[0] for t in transitions]

        obses = np.array([np.array(t[0]) for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)


        
        targets = rewards_t + self.discount_factor * (1 - dones_t) * max_target_q_values




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
