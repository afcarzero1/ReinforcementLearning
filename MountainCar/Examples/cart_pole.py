import os
import pickle
from datetime import datetime

import gym
import numpy as np
from sklearn.model_selection import ParameterGrid
from termcolor import colored

from MountainCar.Modeling.sarsa_modeling import SarsaLambda, AgentTrainer


def define_eta(mode):
    if mode == "keep":
        eta = np.array([[1, 0],
                        [0, 1]])
    elif mode == "combinations":
        eta = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1],
                        ])
    elif mode == "increase_comb":
        eta = np.array([[0, 1],
                        [0, 2],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [2, 0],
                        [2, 1],
                        [2, 2]
                        ])
    elif mode == "full":
        eta = np.array([[0, 0],
                        [0, 1],
                        [0, 2],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [2, 0],
                        [2, 1],
                        [2, 2],
                        ])

    # hidden x dim
    return eta


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

    def grid_search(self, agent_parameters: dict , trainer_parameters : dict = {}):
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


        for hyperparameters in ParameterGrid(agent_parameters):
            for trainer_hyp in ParameterGrid(trainer_parameters):
                i += 1
                print("{:5} / {:5} parameter".format(i, len(ParameterGrid(trainer_parameters)) *len(ParameterGrid(agent_parameters))))

                trainer = self.train_step(hyperparameters,trainer_hyp,verbose=False)
                passed, avg_rew, conf = trainer.test()
                avg_rew_lim = avg_rew - conf



                self.results.append((hyperparameters,trainer_hyp,passed,avg_rew,conf))

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


    def train_step(self, model_parameters : dict, trainer_parameters = {} , verbose = False):
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
        trainer.train(verbose = verbose)

        return trainer


def main():
    # Find best configuration (a)
    env = gym.make('MountainCar-v0')
    env.reset()
    gridsearcher = GridSearcher(env,
                                agent_class=SarsaLambda,
                                agent_trainer_class=AgentTrainer)

    parameters = {
        "eta": [define_eta("increase_comb"), define_eta("full"),
                define_eta("combinations"),define_eta("keep")],
        "discount_factor_gamma": [1., 0.95],
        "lambda_sarsa": [0.1, 0.2, 0.5, 0.7, 0.9],
        "momentum": [0.1, 0.2, 0.5, 0.7, 0.9]
    }

    best_hyperparameters = gridsearcher.grid_search(agent_parameters=parameters)

    # Repeat the training for generating graph (d.1)

    trainer : AgentTrainer = gridsearcher.train_step(best_hyperparameters,verbose=True)

    # Save it
    trainer.agent.save(file_prefix="best",extra_data=best_hyperparameters)


    trainer.play_game()

    # d.2 , d.3
    trainer.agent.plot_value_function()
    trainer.agent.plot_best_action()

    # d.4
    #todo : use the gridearch self.results to see it
    print(gridsearcher.results)

    with open("hyper_search_result","wb") as f:
        pickle.dump(gridsearcher.results,f)

    # e.1

    parameters = { key : [value] for key,value in best_hyperparameters.items()}

    parameters["lambda_sarsa"] = [i/100 for i in range(0,100,5)]
    parameters["discount_factor_gamma"] = [i / 100 for i in range(0, 100, 5)]



    best_hyperparameters = gridsearcher.grid_search(agent_parameters=parameters)

    print(gridsearcher.results)

    with open("discount_search_result","wb") as f:
        pickle.dump(gridsearcher.results,f)

    parameters = {key: [value] for key, value in best_hyperparameters.items()}
    trainer_param = {"learning_rate_initial" : [0.1,0.01,0.05,0.0001,0.00005]}

    best_hyperparameters = gridsearcher.grid_search(agent_parameters=parameters,trainer_parameters=trainer_param)

    print(gridsearcher.results)

    with open("lr_search_result","wb") as f:
        pickle.dump(gridsearcher.results,f)



    # grid_search(parameters)

    # See effect of parameters of the trainer


if __name__ == '__main__':
    main()
