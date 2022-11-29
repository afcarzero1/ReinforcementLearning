import pickle
import os
import gym

from Modeling.sarsa_modeling import GridSearcher
from Modeling.sarsa_modeling import AgentTrainer
from Modeling.sarsa_modeling import SarsaLambda

import matplotlib.pyplot as plt
import numpy as np


def test_different_exploration(file_weights , folder):
    # Repeat training
    rewards = []
    with open(file_weights, "rb") as f:
        results: dict = pickle.load(f)

        print(results["info"])

        env = gym.make('MountainCar-v0')
        env.reset()

        agent = SarsaLambda(2, 3, **results["info"], exploration_strategy="eps")
        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        # Generate image of the training with those parameters
        trainer.train()
        trainer.plot_rewards(os.path.join(folder, "best_rewards_normal.png"))

        eps_rew = trainer.get_training_rewards()

        passed, avg_reward, confidence = trainer.test(N=200, verbose=True)
        rewards.append(("eps",passed,avg_reward,confidence))

        agent = SarsaLambda(2, 3, **results["info"], exploration_strategy="state")
        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        trainer.train()
        state_rew = trainer.get_training_rewards()
        trainer.plot_rewards(os.path.join(folder, "best_rewards_state.png"))
        passed, avg_reward, confidence = trainer.test(N=200, verbose=True)
        rewards.append(("state", passed, avg_reward, confidence))

        agent = SarsaLambda(2, 3, **results["info"], exploration_strategy="opposite")
        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        trainer.train()
        opp_rew = trainer.get_training_rewards()
        print("opp")
        passed, avg_reward, confidence = trainer.test(N=200, verbose=True)
        rewards.append(("opposite", passed, avg_reward, confidence))

        agent = SarsaLambda(2, 3, **results["info"], exploration_strategy="still")
        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        trainer.train()
        still_rew = trainer.get_training_rewards()
        print("still")
        passed, avg_reward, confidence = trainer.test(N=200, verbose=True)
        rewards.append(("still", passed, avg_reward, confidence))

        with open(os.path.join(folder,"different_strategies_results.pkl"),"wb") as f:
            pickle.dump(rewards,f)

        ## PLOT

        ep = np.arange(0, len(state_rew))

        plt.close()

        plt.plot(ep, trainer.running_average(state_rew, 15), label="State Exploration")
        plt.plot(ep, trainer.running_average(eps_rew, 15), label="Standard")
        plt.plot(ep, trainer.running_average(opp_rew, 15), label="Opposite")
        plt.plot(ep, trainer.running_average(still_rew, 15), label="Still")

        plt.grid()

        plt.title("Comparison Different Exploration Strategies")
        plt.legend()

        plt.savefig(os.path.join(folder,"comparison_strategies.png"))
        plt.show()
        plt.close()



def analyze_learning_rate(file_weights , folder):
    ## CREATE ENVIRONMENT
    env = gym.make('MountainCar-v0')
    env.reset()
    gridsearcher = GridSearcher(env,
                                agent_class=SarsaLambda,
                                agent_trainer_class=AgentTrainer)
    ## GET BEST CONFIGURATION

    with open(file_weights, "rb") as f:
        results: dict = pickle.load(f)
        best_hyperparameters = results["info"]

    parameters = {key: [value] for key, value in best_hyperparameters.items()}

    ## TEST LEARNING RATES
    trainer_param = {"learning_rate_initial": [0.01, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                               0.011,0.012,0.013,0.014,0.015,
                                               0.0009,0.0008,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0001]}
    gridsearcher.grid_search(agent_parameters=parameters, trainer_parameters=trainer_param)

    ## ANALYZE THE RESULT
    rewards = []
    confidences = []
    values = []
    for result in gridsearcher.results:
        hyper_trainer = result[1]
        avg_rew = result[3]
        conf = result[4]

        lr = hyper_trainer["learning_rate_initial"]

        values.append(lr)
        rewards.append(avg_rew)
        confidences.append(conf)

    zipped = zip(values, rewards, confidences)
    zipped = sorted(zipped)
    values, rewards, confidences = zip(*zipped)

    ## PLOT THE RESULT
    plt.plot(values, rewards, label="reward")
    plt.legend()
    plt.grid()
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Reward")

    plt.plot(values, np.array(rewards),
             color='blue', marker='o')
    plt.savefig(os.path.join(folder,"learning_rate.png"))
    plt.title("Average Reward")
    plt.show()


def analyze_parameter( parameter_name : str , parameter_values : [] ,file_weights, folder):
    ## CREATE ENVIRONMENT
    env = gym.make('MountainCar-v0')
    env.reset()
    gridsearcher = GridSearcher(env,
                                agent_class=SarsaLambda,
                                agent_trainer_class=AgentTrainer)
    ## GET BEST CONFIGURATION

    with open(file_weights, "rb") as f:
        results: dict = pickle.load(f)
        best_hyperparameters = results["info"]

    parameters = {key: [value] for key, value in best_hyperparameters.items()}

    parameters[parameter_name] = parameter_values
    gridsearcher.grid_search(agent_parameters=parameters)

    ## ANALYZE THE RESULT
    rewards = []
    confidences = []
    values = []
    for result in gridsearcher.results:
        hyper = result[0]
        hyper_trainer = result[1]
        avg_rew = result[3]
        conf = result[4]

        param = hyper[parameter_name]

        values.append(param)
        rewards.append(avg_rew)
        confidences.append(conf)

    zipped = zip(values, rewards, confidences)
    zipped = sorted(zipped)
    values, rewards, confidences = zip(*zipped)

    ## PLOT THE RESULT
    plt.plot(values, rewards, label="reward")
    plt.legend()
    plt.grid()
    plt.xlabel(parameter_name)
    plt.ylabel("Average Reward")

    plt.plot(values, np.array(rewards),
             color='blue', marker='o')
    plt.savefig(os.path.join(folder,parameter_name+".png"))
    plt.title("Average Reward")
    plt.show()


def compare_plots_rewards(parameter_name : str , parameter_values : [], file_weights,folder):
    ## CREATE ENVIRONMENT
    env = gym.make('MountainCar-v0')
    env.reset()
    gridsearcher = GridSearcher(env,
                                agent_class=SarsaLambda,
                                agent_trainer_class=AgentTrainer)
    ## GET BEST CONFIGURATION

    with open(file_weights, "rb") as f:
        results: dict = pickle.load(f)
        best_hyperparameters = results["info"]

    parameters = {key: value for key, value in best_hyperparameters.items()}

    parameters[parameter_name] = parameter_values[0]
    trainer = gridsearcher.train_step(model_parameters=parameters,verbose=True)
    trainer.test(verbose=True)
    trainer.plot_rewards(os.path.join(folder,parameter_name+"1.png"))

    parameters[parameter_name] = parameter_values[1]
    trainer = gridsearcher.train_step(model_parameters=parameters, verbose=True)
    trainer.test(verbose=True)
    trainer.plot_rewards(os.path.join(folder,parameter_name+"2.png"))

