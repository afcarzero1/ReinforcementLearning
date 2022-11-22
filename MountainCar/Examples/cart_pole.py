import os
from typing import List
from datetime import datetime

from MountainCar.Modeling.basis import SarsaLambda


import numpy as np
import gym
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm import trange, tqdm


from sklearn.model_selection import ParameterGrid


class AgentTrainer:
    def __init__(self, environment: gym.Env,
                 agent: SarsaLambda,
                 learning_rate_initial: float = 0.001,
                 epsilon_initial: float = 1,
                 epsilon_decay: str = "linear",
                 number_episodes: int = 500,
                 episode_reward_trigger: float = -150,
                 learning_rate_scaling: List[float] = None,
                 early_stopping=True,
                 information_episodes : int= 50):

        ## SET PARAMETERS
        self.early_stopping = early_stopping
        self.information_episodes = information_episodes
        self.episode_reward_trigger = episode_reward_trigger
        self.number_episodes = number_episodes
        self.env = environment
        self.agent = agent
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_scaling
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay

        if learning_rate_scaling is None:
            self.learning_rate_scaling = [0.5]

        ## SET VARIABLES
        self.episode_reward_list = []

    def _define_epsilon(self):
        if self.epsilon_decay == "linear":
            return np.linspace(self.epsilon_initial, 0, self.number_episodes)
        elif self.epsilon_decay == "exponential":
            powers = np.arange(1, self.number_episodes)
            epsilon: np.ndarray = np.ones(self.number_episodes) * self.epsilon_initial
            return np.power(epsilon, powers)
        else:
            raise NotImplementedError(f"{self.epsilon_decay} mode not implemented")

    def train(self):
        np.random.seed(10)
        ### RESET ENVIRONMENT ###
        self.env.reset()

        ### DEFINE PARAMETERS ###
        NUMBER_EPISODES = self.number_episodes
        LEARNING_RATE_INITIAL = self.learning_rate_initial

        ### DEFINE FUNCTION USEFUL FOR SCALING ###
        def scale_state_variables(s, low=self.env.observation_space.low, high=self.env.observation_space.high):
            ''' Rescaling of s to the box [0,1]^2 '''
            x = (s - low) / (high - low)
            return x

        epsilon = self._define_epsilon()
        self.learning_rate = LEARNING_RATE_INITIAL
        time = 0

        velocity = self.agent.velocity.copy()
        weights = self.agent.weights.copy()

        for e in trange(NUMBER_EPISODES):
            done = False
            terminated = False
            state = scale_state_variables(self.env.reset()[0])
            total_episode_reward = 0.

            while not (done or terminated):
                ### TAKE ACTION ###
                action = self.agent.epsilon_greedy(state, epsilon=epsilon[e])

                ### USE ACTION ###
                next_state, reward, done, terminated, *_ = self.env.step(action)
                next_state = scale_state_variables(next_state, self.env.observation_space.low,
                                                   self.env.observation_space.high)

                ### COMPUTE NEXT ACTION

                next_action = self.agent.epsilon_greedy(next_state, epsilon=epsilon[e])

                ### USE SARSA
                self.agent.forward(state_t=state,
                                   state_t_next=next_state,
                                   action_t=action,
                                   action_t_next=next_action,
                                   reward_t=reward,
                                   learning_rate_t=self.learning_rate)

                # todo : understand why matrix version does not work. Code below helps
                """
                # 1. compute delta
                delta = agent.compute_delta(state_t=state,
                                            state_t_next=next_state,
                                            action_t=action,
                                            action_t_next=next_action,
                                            reward_t=reward)

                # 2. update tracing
                eligibility_trace = agent.update_eligibility_trace(action_t=action, state_t=state)

                # 3. scale lr
                scaled_learning_rate = agent.basis.scale_learning_rate(learning_rate_initial)

                 # 4. update weights
                for index , lr in enumerate(scaled_learning_rate):
                     agent.velocity[:,index] = agent.velocity[:,index] * agent.momentum + lr * delta * agent.eligibility_trace[:,index]
                    agent.weights[:,index] = agent.weights[:,index] + agent.velocity[:,index] * agent.momentum + lr * delta * agent.eligibility_trace[:,index]


                # 4. update weights
                scaled_learning_rate_mat = np.repeat(scaled_learning_rate.reshape(1,agent.hidden_size), repeats=agent.number_actions,
                                                 axis=0)  # (num_action,hidden_size)


                velocity = velocity * agent.momentum + np.multiply(scaled_learning_rate_mat,agent.eligibility_trace) * delta
                weights = weights + velocity * agent.momentum + np.multiply(scaled_learning_rate_mat,agent.eligibility_trace) * delta
                """

                ## UPDATE DATA
                total_episode_reward += reward
                state = next_state

            if total_episode_reward > self.episode_reward_trigger:
                if time < len(self.learning_rate_scaling):
                    self.learning_rate = self.learning_rate_scaling[time] * self.learning_rate
                    time += 1

            self.episode_reward_list.append(total_episode_reward)

            if e % self.information_episodes == 0:
                self.print_episode_information(e)

            if self.moving_average(30) > self.episode_reward_trigger and self.early_stopping:
                self.env.close()
                break

            self.env.close()
        ### PLOT RESULTS ###

        self.plot_rewards()

    def test(self, N=50, verbose = False):
        ### DEFINE FUNCTION USEFUL FOR SCALING ###
        def scale_state_variables(s, low=self.env.observation_space.low, high=self.env.observation_space.high):
            ''' Rescaling of s to the box [0,1]^2 '''
            x = (s - low) / (high - low)
            return x

        N_EPISODES = N  # Number of episodes to run for trainings
        CONFIDENCE_PASS = -135
        print('Checking solution...')
        EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
        episode_reward_list = []
        for i in EPISODES:
            EPISODES.set_description("Episode {}".format(i))
            # Reset enviroment data
            done = False
            truncated = False
            state = scale_state_variables(self.env.reset()[0])
            total_episode_reward = 0.

            action = self.agent.epsilon_greedy(state, epsilon=0)

            while not (done or truncated):
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, truncated, *_ = self.env.step(action)
                next_state = scale_state_variables(next_state)
                next_action = self.agent.epsilon_greedy(state, epsilon=0)

                # Update episode reward
                total_episode_reward += reward

                # Update state for next iteration
                state = next_state
                action = next_action

            # Append episode reward
            episode_reward_list.append(total_episode_reward)

            # Close environment
            self.env.close()

        avg_reward = np.mean(episode_reward_list)
        confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)

        if verbose:
            print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

        if avg_reward - confidence >= CONFIDENCE_PASS:
            if verbose:
                print('Your policy passed the test!')
            return True, avg_reward - confidence
        else:
            if verbose:
                print(
                    'Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(
                        CONFIDENCE_PASS))
            return False, avg_reward - confidence

    def moving_average(self, N=30):
        ep = self.episode_reward_list
        if len(self.episode_reward_list) < N:
            ep = np.zeros_like(self.episode_reward_list)

        running_avg = ep[-N:]
        return np.average(running_avg)

    def print_episode_information(self, e: int):
        running_avg = self.moving_average()
        tqdm.write(" Episode {:5} / {:5} , Reward {:5f} , AvgReward {:5f} , lr : {:5f}".format(e, self.number_episodes,
                                                                                               self.episode_reward_list[
                                                                                                   e],
                                                                                               running_avg,
                                                                                               self.learning_rate))

    def plot_rewards(self):
        plt.plot([i for i in range(1, self.number_episodes + 1)], self.episode_reward_list, label='Episode reward')
        plt.plot([i for i in range(1, self.number_episodes + 1)], self.running_average(self.episode_reward_list, 10),
                 label='Average episode reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')

        plt.suptitle('Total Reward vs Episodes')
        plt.title(str(self.agent))
        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

    def running_average(self, x, N):
        ''' Function used to compute the running mean
            of the last N elements of a vector x
        '''
        if len(x) >= N:
            y = np.copy(x)
            y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
        else:
            y = np.zeros_like(x)
        return y


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
                        [1, 0],
                        [0, 2],
                        [2, 0],
                        [1, 1],
                        [1, 2],
                        [2, 2],
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


def main():
    ### SET GENERAL PARAMETERS
    NUMBER_EPISODES = 500
    DISCOUNT_FACTOR = 1.
    if not os.path.exists(os.path.join(".", "RESULTS")):
        os.makedirs(os.path.join(".", "RESULTS"))

    ### CREATE ENVIRONMENT ###
    env = gym.make('MountainCar-v0')
    env.reset()

    ### CREATE GRID
    parameters = {
        "eta": [define_eta("increase_comb") , define_eta("full")],
        "discount_factor_gamma": [1., 0.95],
        "lambda_sarsa": [0.1, 0.2, 0.5, 0.7, 0.9],
        "momentum": [0.1, 0.2, 0.5, 0.7, 0.9]
    }

    max_reward = float("-inf")
    best_hyperparameters = {}
    stored_time = ""
    i=0
    for hyperparameters in ParameterGrid(parameters):

        print("{:5} / {:5} parameter".format(i,len(ParameterGrid(parameters))))

        ### CREATE LEARNER ###
        agent = SarsaLambda(state_dimension=2,
                            number_actions=env.action_space.n,
                            **hyperparameters)

        trainer = AgentTrainer(environment=env,
                               agent=agent,
                               number_episodes=NUMBER_EPISODES,
                               episode_reward_trigger=-135,
                               epsilon_initial=0.8,
                               early_stopping=False,
                               information_episodes=1000)

        trainer.test()

        trainer.train()

        passed, avg_rew_lim = trainer.test()
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

    print(colored("The best policy, stored at time " + stored_time + " is:", 'red'))
    print(best_hyperparameters)


if __name__ == '__main__':
    main()
