import pickle
import os
import gym

from MountainCar.Examples.cart_pole import AgentTrainer, GridSearcher
from MountainCar.Modeling.sarsa_modeling import SarsaLambda

import matplotlib.pyplot as plt
import numpy as np

def main(file_name):

    folder = "BEST_RESULTS"

    if not os.path.exists(folder):
        os.makedirs(folder)


    with open("RESULTS/best_weights.pkl", "rb") as f:
        results: dict = pickle.load(f)

        print(results["info"])

        env = gym.make('MountainCar-v0')
        env.reset()

        agent = SarsaLambda(2, 3,**results["info"],exploration_strategy="eps")
        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        # Generate image of the training with those parameters
        trainer.train()
        trainer.plot_rewards(os.path.join(folder,"best_rewards_normal.png"))

        eps_rew = trainer.get_training_rewards()


        plt.plot()

        trainer.test(N=200,verbose=True)

        agent = SarsaLambda(2, 3, **results["info"], exploration_strategy="state")
        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        trainer.train()
        state_rew = trainer.get_training_rewards()
        trainer.plot_rewards(os.path.join(folder, "best_rewards_opposite.png"))
        trainer.test(N=200, verbose=True)

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
        trainer.test(N=200, verbose=True)

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
        trainer.test(N=200, verbose=True)




        ## PLOT

        ep = np.arange(0,len(state_rew))

        plt.close()

        plt.plot(ep,trainer.running_average(state_rew,15),label = "State Exploration")
        plt.plot(ep,trainer.running_average(eps_rew,15),label = "Standard")
        plt.plot(ep, trainer.running_average(opp_rew, 15), label="Opposite")
        plt.plot(ep, trainer.running_average(still_rew, 15), label="Still")

        plt.grid()

        plt.title("Comparison Different Exploration Strategies")
        plt.legend()

        plt.savefig("comparison_strategies")
        plt.show()


    with open("best_agent.pkl", "rb") as f:
        env = gym.make('MountainCar-v0', render_mode="human")
        env.reset()
        agent2: SarsaLambda = pickle.load(f)

        agent2.plot_best_action(os.path.join(folder,"best_actions.png"))
        agent2.plot_value_function(os.path.join(folder,"best_values.png"))
        trainer = AgentTrainer(environment=env,
                               agent=agent2)
        trainer.play_game()


if __name__ == '__main__':
    main("goodone_weights.pkl")
