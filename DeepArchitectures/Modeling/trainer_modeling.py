import os
import pickle
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, List

import gym
import numpy as np
from sklearn.model_selection import ParameterGrid
from termcolor import colored
from tqdm import trange, tqdm

import matplotlib.pyplot as plt

import copy


class Agent(ABC):
    @abstractmethod
    def forward(self, state: Any, epsilon: float = 0) -> Any:
        """
        Agent that takes a decision given an action with an epsilon greedy policy.

        Args:
            state (Any): The state from which decision has to be taken.
            epsilon (float): The epsilon to use for the decision
        Return:
            action (Any): The action to be taken
        """
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    def save(self, path, extra_data) -> None:
        pass


class AgentEpisodicTrainer(ABC):
    r"""
    Base trainer class using episode buffer.
    """

    def __init__(self, environment: gym.Env,
                 agent: Agent,
                 learning_rate_initial: float = 0.001,
                 discount_factor: float = 1,
                 epsilon_initial: float = 1,
                 epsilon_decay: str = "linear",
                 number_episodes: int = 500,
                 learning_rate_scaling: List[float] = None,
                 early_stopping=False,
                 early_stopping_trigger: float = -150,
                 early_stopping_episodes_trigger: int = 50,
                 information_episodes: int = 50,
                 buffer_size: int = 100,
                 buffer_size_min: int = 30,
                 batch_size: int = 30):
        r"""
        Initialize the trainer.

        Args:
            environment (gym.Env) : The environment to use for training.
            agent (Agent) : The agent to be trained.
            learning_rate_initial (float) : The initial learning rate to train the agent.
            discount_factor (float) : The discount factor to be used for training.
            epsilon_initial (float) : The initial epsilon to be used in the exploration
            epsilon_decay (str) : The trype of epsilon decay. It migh be "linear" or "exponential"
            number_episodes (int) : The number of episodes to be used in training
            learning_rate_scaling (List[float]) : The learning rate scale to be used
            early_stopping (bool) : Wether to stop the training when early_stopping_trigger average reward is reached.
            early_stopping_trigger (float) : Threshold for stopping training
            early_stopping_episodes_trigger (int) : Number of episodes on which the average reward is computed for the
                early stopping
            information_episodes (int) : Number of episodes after which information is printed
            buffer_size (int) : The sieze of the buffer
            buffer_size_min (int) : The minimum size of the buffer (filled with random experiences at the beginning)
            batch_size (int) : The size of the batch on which the agent is updated.
            """

        ## SET PARAMETERS
        self.step = 0
        self.early_stopping = early_stopping
        self.information_episodes = information_episodes
        self.episode_reward_trigger = early_stopping_trigger
        self.early_stopping_episodes_trigger = early_stopping_episodes_trigger
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
        self.episode_step_list = []
        self.replay_buffer = deque(maxlen=buffer_size)
        self.min_buffer_size = buffer_size_min
        self.batch_size = batch_size
        assert buffer_size_min <= buffer_size
        assert buffer_size >= 0

    def _define_epsilon(self):
        if self.epsilon_decay == "linear":
            return np.linspace(self.epsilon_initial, 0, self.number_episodes)
        elif self.epsilon_decay == "exponential":
            if self.epsilon_initial == 1:
                self.epsilon_initial = 0.999
            powers = np.arange(0, self.number_episodes)
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
        ### RESET ENVIRONMENT ###
        self.env.reset()
        epsilon = self._define_epsilon()
        self.learning_rate = self.learning_rate_initial
        self._initialize_replay_buffer()

        self.step = 0
        self.episode = 0
        self.episode_step_list = []
        self.episode_reward_list = []
        for e in trange(self.number_episodes):
            self.episode = e
            done = False
            terminated = False
            state = self.env.reset()[0]
            total_episode_reward = 0.

            while not (done or terminated):
                ### TAKE ACTION ###

                action = self.action_agent_callback(state, epsilon[e])

                ### USE ACTION ###
                next_state, reward, done, terminated, *_ = self.env.step(action)

                ### ADD TO REPLAY BUFFER ###
                self.replay_buffer.append((state, action, reward, done, next_state))

                ## UPDATE DATA
                total_episode_reward += reward
                state = next_state

                ### UPDATE OF NETWORK ###

                self.update_agent()

                self.step += 1

            self.episode_reward_list.append(total_episode_reward)
            self.episode_step_list.append(self.step)

            if e % self.information_episodes == 0:
                self.print_episode_information(e)

            if self._moving_average(
                    self.early_stopping_episodes_trigger) > self.episode_reward_trigger and self.early_stopping:
                self.env.close()
                break

            self.episode_finished_callback()

        ### PLOT RESULTS ###
        if verbose:
            self.plot_rewards()
            self.plot_steps()

    def test(self, N=50, verbose=False):
        N_EPISODES = N  # Number of episodes to run for trainings
        CONFIDENCE_PASS = self.episode_reward_trigger
        EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
        episode_reward_list = []
        for i in EPISODES:
            EPISODES.set_description("Episode {}".format(i))
            # Reset enviroment data
            done = False
            truncated = False
            state = self.env.reset()[0]
            total_episode_reward = 0.

            action = self.action_agent_callback(state, epsilon=0)

            while not (done or truncated):
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, truncated, *_ = self.env.step(action)
                next_action = self.action_agent_callback(state, epsilon=0)

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
        action = self.action_agent_callback(state, epsilon=0)

        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, *_ = self.env.step(action)
            next_action = self.action_agent_callback(next_state, epsilon=0)

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
        plt.plot([i for i in range(1, len(self.episode_reward_list) + 1)], self.episode_reward_list,
                 label='Episode reward')
        plt.plot([i for i in range(1, len(self.episode_reward_list) + 1)],
                 self.running_average(self.episode_reward_list, 10),
                 label='Average episode reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total reward')

        plt.suptitle('Total Reward vs Episodes')
        # plt.title(str(self.agent))
        plt.legend()
        plt.grid(alpha=0.3)

        if file_name == "":
            plt.show()
            plt.close()
        else:
            plt.savefig(file_name)
            plt.close()

    def plot_steps(self,file_name: str = ""):
        plt.plot([i for i in range(1, len(self.episode_step_list) + 1)], self.episode_step_list,
                 label='Episode Steps')
        plt.plot([i for i in range(1, len(self.episode_step_list) + 1)],
                 self.running_average(self.episode_step_list, 10),
                 label='Average episode steps')
        plt.xlabel('Episodes')
        plt.ylabel('Total steps')

        plt.suptitle('Total steps vs Episodes')
        # plt.title(str(self.agent))
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

    def action_agent_callback(self, state: Any, epsilon: float):
        return self.agent.forward(state=state, epsilon=epsilon)

    def episode_finished_callback(self):
        pass


class BigHyperParameter:
    r"""
    Wrapper class for parameters that cannot be directly printed or saved.
    """

    def __init__(self, param: Any, name: str):
        self._param = param
        self._name = name

    def value(self):
        return self._param

    def name(self):
        return self._name

    def __str__(self):
        return self._name


class GridSearcher:
    r"""
    Class for performing a grid search among the parameters of an agent class and a trainer class.
    """

    def __init__(self, env: gym.Env, agent_class, agent_trainer_class, save_all=False, folder="RESULTS"):
        r"""
        Initialize the grid searcher

        Args:
            env (gym.Env) : The environment to test
            agent_class (Type) : The type of the agent class
            agent_trainer_class (Type) : The type of the trainer class
            save_all (bool) : Save all the results
            folder (str) : The path to save the results
        """
        self.env = env
        self.agent_class = agent_class
        self.agent_trainer_class = agent_trainer_class
        self.folder = folder
        self.save_all = save_all

        self.results = []

    def grid_search(self, agent_parameters: dict, trainer_parameters: dict = {}):
        r"""
        Perform the grid search.

        Args:
            agent_parameters (dict) : Dictionary with the paramaters to test
            trainer_parameters (dict) : Dictionary with the trainer parameters to test
        """
        ### SET GENERAL PARAMETERS
        e = datetime.now()
        time0: str = e.strftime("%Y-%m-%d%H-%M-%S")
        if not os.path.exists(os.path.join(".", self.folder, time0)):
            os.makedirs(os.path.join(".", self.folder, time0))

        self.folder_results = os.path.join(self.folder, time0)

        ### CREATE ENVIRONMENT ###
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

                # Save the results in a printable/savable manner
                self.results.append((self._unwrap_name_dict(hyperparameters),
                                     self._unwrap_name_dict(trainer_hyp),
                                     passed,
                                     avg_rew,
                                     conf)
                                    )
                print(f"Average reward : {avg_rew} +- {conf}")

                ### IF PASSED THE TEST SAVE THE MODEL AND HYPERPARAMETERS ###
                e = datetime.now()
                time = e.strftime("%Y-%m-%d%H-%M-%S")
                if self.save_all:
                    trainer.agent.save(os.path.join(self.folder, time0, f'{i:03}'),
                                       extra_data={"agent_param": self._unwrap_name_dict(hyperparameters),
                                                   "trainer_param": self._unwrap_name_dict(trainer_hyp),
                                                   "result": (passed, avg_rew, conf)})

                    trainer.plot_rewards(os.path.join(self.folder, time0, f'{i:03}' + ".png"))
                    trainer.plot_steps(os.path.join(self.folder, time0, f'{i:03}steps' + ".png"))
                if passed:
                    if not self.save_all:
                        e = datetime.now()
                        time = e.strftime("%Y-%m-%d%H-%M-%S")

                        trainer.agent.save(os.path.join(self.folder, time0, f'{i:03}'),
                                           extra_data={"agent_param": self._unwrap_name_dict(hyperparameters),
                                                       "trainer_param": self._unwrap_name_dict(trainer_hyp),
                                                       "result": (passed, avg_rew, conf)})

                        trainer.plot_rewards(os.path.join(self.folder, time0, f'{i:03}' + ".png"))
                        trainer.plot_steps(os.path.join(self.folder, time0, f'{i:03}steps' + ".png"))

                    if avg_rew_lim > max_reward:
                        print(colored("[NEW BEST] The new best hyperparameter combination is:"))
                        print(hyperparameters)
                        max_reward = avg_rew_lim
                        best_hyperparameters = (hyperparameters, trainer_hyp)
                        stored_time = time

        ### RETURN BEST FOUND POLICY
        print(colored("The best policy, stored at time " + stored_time + " is:", 'red'))
        print(best_hyperparameters)
        return best_hyperparameters

    def train_step(self, model_parameters: dict, trainer_parameters: dict = {}, verbose=False):
        r"""
        Do one iteration of training of the agent

        Args:
            model_parameters (dict) : Dictionary with parameters of the agent.
            trainer_parameters (dict) : Dictionary with the parameters of the trainer.
            verbose (bool) : Display results in terminal.

        Returns:
            trainer (Any) : The trainer instance resulting after the training.
        """

        agent = self.agent_class(**self._unwrap_dict(model_parameters))
        ### CREATE TRAINER ###
        trainer = self.agent_trainer_class(agent=agent, **self._unwrap_dict(trainer_parameters))

        ### TRAIN AND TEST ###
        trainer.train(verbose=verbose)

        return trainer

    def _unwrap_name_dict(self, parameter_set: dict) -> dict:
        return {k: self._unwrap_name(v) for k, v in parameter_set.items()}

    def _unwrap_name(self, parameter: Any):
        if isinstance(parameter, BigHyperParameter):
            return parameter.name()
        else:
            return parameter

    def _unwrap_dict(self, parameter_set: dict) -> dict:
        return {k: self._unwrap(v) for k, v in parameter_set.items()}

    def _unwrap(self, parameter: Any) -> Any:
        r"""
        Take out the parameter value
        """
        if isinstance(parameter, BigHyperParameter):
            return parameter.value()
        else:
            return parameter

    @staticmethod
    def analyze_results(directory):
        r"""
        Analyze the results in one directory to find the best configuration and print it.
        """
        files = Path(directory).glob('*')

        best_conf_name = ""
        best_conf = None
        best_result = float("-inf")
        for file in files:

            # Open the results
            if str(file).lower().endswith(".pkl"):
                with open(file, "rb") as f:
                    results = pickle.load(f)

                    passed, avg_rew, conf = results["result"]

                    if avg_rew - conf > best_result:
                        best_conf_name = str(file)
                        best_conf = results
                        best_result = avg_rew - conf

        best_agent_param: dict = best_conf["agent_param"]
        best_trainer_param: dict = best_conf["trainer_param"]

        print("BEST AGENT PARAMETER")
        print(best_agent_param)
        print("NEST TRAINER PARMATER")
        print(best_trainer_param)
        print("RESULTS")
        print(best_conf["result"])
        print(best_conf_name)

        return best_agent_param,best_trainer_param

    @staticmethod
    def compare_feature(directory, agent=False, param_name="learning_rate_initial", log_scale=False):
        files = Path(directory).glob('*')
        params_results = []
        params_values = []
        for file in files:
            best_conf_name = ""
            best_conf = None
            best_result = float("-inf")
            # Open the results

            if str(file).lower().endswith(".pkl"):
                with open(file, "rb") as f:
                    results = pickle.load(f)

                    passed, avg_rew, conf = results["result"]

                    params_values.append(results["trainer_param" if not agent else "agent_param"][param_name])
                    params_results.append(avg_rew)

                    if avg_rew - conf > best_result:
                        best_conf_name = str(file)
                        best_conf = results
                        best_result = avg_rew - conf

        zipped = zip(params_values, params_results)
        zipped = sorted(zipped)
        params_values = [np.log10(x) if log_scale else x for x, _ in zipped]
        params_results = [x for _, x in zipped]

        plt.plot(params_values, params_results, 'bo-')
        plt.title(param_name)
        plt.xlabel("Value" if not log_scale else "log(Value)")
        plt.ylabel("Average Reward")
        plt.show()
