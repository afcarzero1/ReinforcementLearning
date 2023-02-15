import json
import os
import pickle
from pathlib import Path
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from Modeling.trainer_modeling import GridSearcher, BigHyperParameter
from Modeling.dqn_modeling import  AgentDQN, AgentEpisodicDQNTrainer
from Modeling.NetworkModeling import SmallNetwork,BigNetwork



def solve_problem():
    env = gym.make('LunarLander-v2')
    env.reset()

    q_learner_small = SmallNetwork(env)

    agent = AgentDQN(network=q_learner_small,
                     env=env,
                     network_initialization_parameters={"env": env})

    trainer = AgentEpisodicDQNTrainer(env,
                                      agent,
                                      discount_factor=0.99,
                                      learning_rate_initial=5e-4,
                                      batch_size=64,
                                      buffer_size=5000,
                                      buffer_size_min=64,
                                      early_stopping=True,
                                      early_stopping_trigger=200,
                                      early_stopping_episodes_trigger=50)

    trainer.train()
    trainer.plot_rewards()
    trainer.test(verbose=True, N=100)
    trainer.agent.save("network.pth", extra_data=None)
    env = gym.make('LunarLander-v2', render_mode="human")
    trainer = AgentEpisodicDQNTrainer(env, agent)
    trainer.play_game()


def find_best_hyperparameters():
    env = gym.make('LunarLander-v2')
    grid_searcher = GridSearcher(env, agent_class=AgentDQN, agent_trainer_class=AgentEpisodicDQNTrainer, save_all=True)

    q_learner_small = SmallNetwork(env)
    q_learner_big = BigNetwork(env)

    q_learner_small = BigHyperParameter(q_learner_small,
                                        "small")  # Create a Hyperparameter that is not supposed to be saved explicitly
    q_learner_big = BigHyperParameter(q_learner_big, "big")

    agent_parameters = {"network": [q_learner_big,q_learner_small],
                        "env": [env],
                        "network_initialization_parameters": [{"env": env}]}

    trainer_parameters = {"discount_factor": [0.99,0.9],
                          "learning_rate_initial": [5e-4, 1e-4, 7e-4],
                          "batch_size": [256],
                          "buffer_size": [5000, 25000, 50000,100000],
                          "buffer_size_min": [256],
                          "early_stopping": [True],
                          "target_update_frequency": [200],
                          "target_update_strategy": ["step"],
                          "early_stopping_trigger": [200],
                          "early_stopping_episodes_trigger": [50],
                          "environment": [env],
                          "epsilon_decay": ["exponential", "linear"],
                          }

    best_hyp = grid_searcher.grid_search(agent_parameters, trainer_parameters)

    print(grid_searcher.results)
    with open("results", "wb") as f:
        pickle.dump(grid_searcher.results, f)

    return grid_searcher.folder_results


def test_network(path):
    env = gym.make('LunarLander-v2')
    env.reset()

    q_learner_small = SmallNetwork(env)

    agent = AgentDQN(network=q_learner_small,
                     env=env,
                     network_initialization_parameters={"env": env})

    agent.load(path)

    agent.plot_q(device="cuda")
    agent.plot_a(device="cuda")

    trainer = AgentEpisodicDQNTrainer(env,
                                      agent,
                                      discount_factor=0.99,
                                      learning_rate_initial=5e-4,
                                      batch_size=64,
                                      buffer_size=5000,
                                      buffer_size_min=64,
                                      early_stopping=True,
                                      early_stopping_trigger=200,
                                      early_stopping_episodes_trigger=50)

    trainer.test(verbose=True, N=200)


    ## PLAY A GAME ##
    env = gym.make('LunarLander-v2', render_mode="human")
    trainer = AgentEpisodicDQNTrainer(env, agent)
    trainer.play_game()


def analyze_results(directory):
    files = Path(directory).glob('*')

    best_conf_name = ""
    best_conf = None
    best_result = float("-inf")
    for file in files:

        # Open the results
        if str(file).lower().endswith(".pkl"):
            with open(file,"rb") as f:
                results=pickle.load(f)

                passed, avg_rew , conf = results["result"]

                if avg_rew - conf > best_result:
                    best_conf_name = str(file)
                    best_conf = results
                    best_result = avg_rew - conf


    best_agent_param : dict = best_conf["agent_param"]
    best_trainer_param : dict = best_conf["trainer_param"]

    print(best_agent_param)
    print(best_trainer_param)
    print(best_conf["result"])
    print(best_conf_name)

    ## LOAD THE AGENT ##

    # Build correct file name
    head = os.path.split(best_conf_name)[0]
    tail = os.path.split(best_conf_name)[1]
    tail = tail.replace("extra.pkl","")
    network = tail + "network.pth"
    path = os.path.join(head,network)

    test_network(path)

    ## RETRAIN CHANGING DISCOUNT FACTOR ##

    grid_searcher = GridSearcher(best_agent_param["env"], agent_class=AgentDQN,
                                 agent_trainer_class=AgentEpisodicDQNTrainer, save_all=True,
                                 folder="NumberEpisodesRESULTS")

    q_learner_small = SmallNetwork(best_agent_param["env"])
    q_learner_big = BigNetwork(best_agent_param["env"])

    ## NUMBER EPISODES
    agent_param = {}
    for k,v in best_agent_param.items():
        if k == "network":
            agent_param[k] = [q_learner_small] if v == "small" else [q_learner_big]
        else:
            agent_param[k]=[v]

    trainer_param = {k : [v,50,100,1000] if k == "number_episodes" else [v] for k,v in best_trainer_param.items()}
    trainer_param["number_episodes"]= [50,100,500,1000]
    grid_searcher.grid_search(agent_param, trainer_param)

    agent_param = {}
    for k,v in best_agent_param.items():
        if k == "network":
            agent_param[k] = [q_learner_small] if v == "small" else [q_learner_big]
        else:
            agent_param[k]=[v]

    trainer_param = {k : [v,1,0.1] if k == "discount_factor" else [v] for k,v in best_trainer_param.items()}

    grid_searcher.grid_search(agent_param,trainer_param)

    ## INVESTIGATE EFFECT OF CHANGING MEMORY ##

    grid_searcher = GridSearcher(best_agent_param["env"], agent_class=AgentDQN,
                                 agent_trainer_class=AgentEpisodicDQNTrainer, save_all=True,
                                 folder="MemoryEffectRESULTS")


    agent_param = {}
    for k,v in best_agent_param.items():
        if k == "network":
            agent_param[k] = [q_learner_small] if v == "small" else [q_learner_big]
        else:
            agent_param[k]=[v]

    trainer_param = {k: [v,500,5000,100000] if k == "buffer_size" else [v] for k, v in best_trainer_param.items()}

    grid_searcher.grid_search(agent_param, trainer_param)

#todo : remove this function. new version in the grid searcher class
def compare_feature(directory , agent= False, param_name = "learning_rate_initial", log_scale = False):
    files = Path(directory).glob('*')
    params_results = []
    params_values = []
    for file in files:
        best_conf_name = ""
        best_conf = None
        best_result = float("-inf")
        # Open the results



        if str(file).lower().endswith(".pkl"):
            with open(file,"rb") as f:
                results=pickle.load(f)

                passed, avg_rew , conf = results["result"]

                params_values.append(results["trainer_param" if not agent else "agent_param"][param_name])
                params_results.append(avg_rew)


                if avg_rew - conf > best_result:
                    best_conf_name = str(file)
                    best_conf = results
                    best_result = avg_rew - conf


    zipped = zip(params_values,params_results)
    zipped = sorted(zipped)
    params_values = [np.log10(x) if log_scale else x for x,_ in zipped]
    params_results = [x for _,x in zipped]

    plt.plot(params_values,params_results,'bo-')
    plt.title(param_name)
    plt.xlabel("Value" if not log_scale else "log(Value)")
    plt.ylabel("Average Reward")
    plt.show()


def re_save_network(path_actor):
    env = gym.make('LunarLander-v2')
    env.reset()
    q_learner_small = SmallNetwork(env)

    agent = AgentDQN(network=q_learner_small,
                     env=env,
                     network_initialization_parameters={"env": env})

    agent.load(path_actor)
    agent.to("cpu")

    #agent.save("best",extra_data=None,dictionary=False)

    torch.save(agent.online_network,"neural-network-1.pth")


def main():
    re_save_network("/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/RESULTS/2022-12-0311-51-18/090network.pth")
    #compare_feature("/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/LearningRateEffectRESULTS/2022-12-0401-10-05",
    #               agent=False,
    #               param_name="discount_factor",
    #               log_scale=True)

    #compare_feature(
    #    "/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/MemoryEffectRESULTS/2022-12-0310-37-16",
     #   agent=False,
     #   param_name="buffer_size",
     #   log_scale=False)
    #analyze_results("/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/RESULTS/2022-12-0222-37-56/")
    # test_network("Modeling/RESULTS_LABORATORY/network.pth")
    #test_network(
    #    "/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/RESULTS/2022-12-0222-37-56/018network.pth")
    #folder = find_best_hyperparameters()
    #analyze_results("/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/RESULTS/2022-12-0311-51-18")
    # solve_problem()

    #gs = GridSearcher.compare_feature("/home/andres/Documents/Master/ReinforcementLearning/ReinforcementLearning/DQN/NumberEpisodesRESULTS/2022-12-1521-17-57",
    #                                  agent=False,param_name="number_episodes")

if __name__ == '__main__':
    main()
