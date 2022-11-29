from abc import ABC, abstractmethod
from collections import deque
from typing import Any, List

import gym
import numpy as np
from tqdm import trange, tqdm

import matplotlib.pyplot as plt


class Agent(ABC):
    @abstractmethod
    def forward(self, state: Any, epsilon: float = 0) -> Any:
        """
        Agent that takes a decision given an action with an epsilon greedy policy.

        Args:
            state: The state
            epsilon (float): The epslion to use
        Return:
            action (Any): The action to be taken
        """
        pass

    @abstractmethod
    def backward(self):
        pass


class AgentEpisodicTrainer(ABC):
    def __init__(self, environment: gym.Env,
                 agent: Agent,
                 learning_rate_initial: float = 0.001,
                 discount_factor : float = 1,
                 epsilon_initial: float = 1,
                 epsilon_decay: str = "linear",
                 number_episodes: int = 500,
                 episode_reward_trigger: float = -150,
                 learning_rate_scaling: List[float] = None,
                 early_stopping=True,
                 information_episodes: int = 50,
                 buffer_size: int = 100,
                 buffer_size_min: int = 30,
                 batch_size: int = 30):

        ## SET PARAMETERS
        self.early_stopping = early_stopping
        self.information_episodes = information_episodes
        self.episode_reward_trigger = episode_reward_trigger
        self.number_episodes = number_episodes
        self.env = environment
        self.agent = agent
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_scaling
        self.discount_factor = discount_factor
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay

        if learning_rate_scaling is None:
            self.learning_rate_scaling = [0.5]

        ## SET VARIABLES
        self.episode_reward_list = []
        self.replay_buffer = deque(maxlen=buffer_size)
        self.min_buffer_size = buffer_size_min
        self.batch_size = batch_size
        assert buffer_size_min <= buffer_size
        assert buffer_size >= 0

    def _define_epsilon(self):
        if self.epsilon_decay == "linear":
            return np.linspace(self.epsilon_initial, 0, self.number_episodes)
        elif self.epsilon_decay == "exponential":
            powers = np.arange(1, self.number_episodes)
            epsilon: np.ndarray = np.ones(self.number_episodes) * self.epsilon_initial
            return np.power(epsilon, powers)
        else:
            raise NotImplementedError(f"{self.epsilon_decay} mode not implemented")

    def _initialize_replay_buffer(self):
        """
        Initialize the replay buffer with random experiences. The replay buffer is replaced and filled with
        min_buffer_size experiences.
        """
        # Drop last replay buffer (if any)
        self.replay_buffer = deque(maxlen=self.replay_buffer.maxlen)

        # Fill it with random experiences
        obs = self.env.reset()[0]
        for _ in range(self.min_buffer_size):
            action = self.env.action_space.sample()
            new_obs, rew, done, truncated, *_ = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            # Reset in case it is necessary
            if done or truncated:
                obs = self.env.reset()[0]

    def train(self, verbose=False):
        # np.random.seed(10)
        ### RESET ENVIRONMENT ###
        self.env.reset()
        epsilon = self._define_epsilon()
        self.learning_rate = self.learning_rate_initial
        self._initialize_replay_buffer()

        for e in trange(self.number_episodes):
            done = False
            terminated = False
            state = self.env.reset()[0]
            total_episode_reward = 0.

            while not (done or terminated):
                ### TAKE ACTION ###
                action = self.agent.forward(state=state, epsilon=epsilon[e])

                ### USE ACTION ###
                next_state, reward, done, terminated, *_ = self.env.step(action)

                ### ADD TO REPLAY BUFFER ###
                self.replay_buffer.append((state, action, reward, done, next_state))

                ## UPDATE DATA
                total_episode_reward += reward
                state = next_state

                ### UPDATE OF NETWORK ###

                self.update_agent()

            self.episode_reward_list.append(total_episode_reward)

            if e % self.information_episodes == 0:
                self.print_episode_information(e)

            if self._moving_average(30) > self.episode_reward_trigger and self.early_stopping:
                self.env.close()
                break

        ### PLOT RESULTS ###
        if verbose:
            self.plot_rewards()

    def test(self, N=50, verbose=False):
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
            state = self.env.reset()[0]
            total_episode_reward = 0.

            action = self.agent.forward(state, epsilon=0)

            while not (done or truncated):
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, truncated, *_ = self.env.step(action)
                next_action = self.agent.forward(state, epsilon=0)

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
            return True, avg_reward, confidence
        else:
            if verbose:
                print(
                    'Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(
                        CONFIDENCE_PASS))
            return False, avg_reward, confidence

    def play_game(self):
        done = False
        truncated = False
        total_episode_reward = 0

        state = self.env.reset()[0]
        action = self.agent.forward(state, epsilon=0)

        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, *_ = self.env.step(action)
            next_action = self.agent.forward(state, epsilon=0)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            action = next_action

            self.env.render()

    def get_training_rewards(self):
        return self.episode_reward_list

    def _moving_average(self, N=30):
        ep = self.episode_reward_list
        if len(self.episode_reward_list) < N:
            ep = np.zeros_like(self.episode_reward_list)

        running_avg = ep[-N:]
        return np.average(running_avg)

    def print_episode_information(self, e: int):
        running_avg = self._moving_average()
        tqdm.write(" Episode {:5} / {:5} , Reward {:5f} , AvgReward {:5f} , lr : {:5f}".format(e, self.number_episodes,
                                                                                               self.episode_reward_list[
                                                                                                   e],
                                                                                               running_avg,
                                                                                               self.learning_rate))

    def plot_rewards(self, file_name: str = ""):
        plt.plot([i for i in range(1, self.number_episodes + 1)], self.episode_reward_list, label='Episode reward')
        plt.plot([i for i in range(1, self.number_episodes + 1)], self.running_average(self.episode_reward_list, 10),
                 label='Average episode reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')

        plt.suptitle('Total Reward vs Episodes')
        plt.title(str(self.agent))
        plt.legend()
        plt.grid(alpha=0.3)

        if file_name == "":
            plt.show()
            plt.close()
        else:
            plt.savefig(file_name)
            plt.close()

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

    @abstractmethod
    def update_agent(self):
        pass
