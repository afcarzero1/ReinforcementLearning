import os
import pickle

import gym
from termcolor import colored

from result_analyzer import test_different_exploration, analyze_learning_rate, analyze_parameter, compare_plots_rewards
from Modeling.sarsa_modeling import SarsaLambda, AgentTrainer, define_eta, GridSearcher

import matplotlib.pyplot as plt


def point_b(grid_searcher):
    r"""
    Find the best configuration of parameters
    """
    # Find best configuration (a)
    parameters = {
        "eta": [define_eta("increase_comb"), define_eta("full"),
                define_eta("keep"), define_eta("combinations")],
        "discount_factor_gamma": [1., 0.95],
        "lambda_sarsa": [0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9],
        "momentum": [0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9]
    }

    return grid_searcher.grid_search(agent_parameters=parameters), grid_searcher


def point_d(best_hyperparameters, gridsearcher, results_folder):
    # Repeat the training for generating graph (d.1)
    trainer: AgentTrainer = gridsearcher.train_step(best_hyperparameters, verbose=True)

    trainer.plot_rewards(os.path.join(results_folder, "best_rewards.png"))

    # d.2 , d.3
    trainer.agent.plot_value_function(os.path.join(results_folder, "best_values.png"))
    trainer.agent.plot_best_action(os.path.join(results_folder, "best_actions.png"))

    return trainer.agent


def point_e(weights_path, folder):
    test_different_exploration(weights_path, folder)

    analyze_learning_rate(weights_path, folder)
    analyze_parameter("lambda_sarsa", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], weights_path, folder)
    analyze_parameter("momentum", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], weights_path, folder)

    compare_plots_rewards("eta", [define_eta("increase_comb"), define_eta("full")], weights_path, folder)
    compare_plots_rewards("initialization", ["random","He"], weights_path, folder)


def main():
    # PARAMETERS
    results_folder = "RESULTS_LABORATORY"
    print(colored(f"Results will be saved in {results_folder}", 'red'))

    # Find best configuration (a , b)
    env = gym.make('MountainCar-v0')
    env.reset()
    grid_searcher = GridSearcher(env,
                                 agent_class=SarsaLambda,
                                 agent_trainer_class=AgentTrainer)

    best_hyper_parameters, grid_searcher = point_b(grid_searcher)

    # Save results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(os.path.join(results_folder, "best_hyper_parameters.pkl"), "wb") as f:
        pickle.dump(best_hyper_parameters, f)

    with open(os.path.join(results_folder, "hyper_search_result.pkl"), "wb") as f:
        pickle.dump(grid_searcher.results, f)

    # Repeat the training for generating graph of training (d.1)
    best_agent = point_d(best_hyper_parameters, grid_searcher, results_folder)

    plt.close('all')

    # Save agent weights and the agent itself. Save also its training parameters.
    best_agent.save(folder=results_folder, extra_data=best_hyper_parameters)

    # Point e
    # Test different exploration strategies
    wights_file = os.path.join(results_folder, "weights.pkl")

    point_e(wights_file,results_folder)


def read_results():
    with open("RESULTS_LABORATORY/best_hyper_parameters.pkl", "rb") as f:
        results = pickle.load(f)
        print(results)

    with open("RESULTS_LABORATORY/hyper_search_result.pkl", "rb") as f:
        hyper_res = pickle.load(f)
        import numpy as np
        eta_r = results.pop("eta")
        for line in hyper_res:

            eta = line[0].pop("eta")

            if np.array_equal(eta, eta_r) and line[0] == results:
                print(line)

    with open("RESULTS_LABORATORY/different_strategies_results.pkl", "rb") as f:
        results = pickle.load(f)
        print(results)


if __name__ == '__main__':
    #point_e("RESULTS_LABORATORY/weights.pkl","RESULTS_LABORATORY")
    main()
    read_results()
