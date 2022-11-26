"""
Model for an agent learning with a SARSA algorithm with linear approximation as function approximator. It uses a Fourier
basis
"""
from datetime import datetime
import os
from abc import ABC, abstractmethod
from typing import Union, Any, List

import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

from matplotlib import cm
from sklearn.model_selection import ParameterGrid
from termcolor import colored
from tqdm import tqdm, trange


class Basis(ABC):
    r"""
    Generic basis to be applied on vectors.
    """

    def __init__(self, input_size: int, output_size: int):
        r"""
        Initialize the basis

        Args:
            input_size(int): Dimension of the starting space
            output_size(int): Dimension of the ending space
        """
        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def to_basis(self, vector: np.ndarray):
        r"""
        Transform the vector to the given basis

        Args:
            vector (np.ndarray) : vector to transform
        """
        pass

    @abstractmethod
    def __call__(self, vector: np.ndarray):
        pass

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._output_size


class FourierBasis(Basis):
    def __init__(self, input_size: int, output_size: int, eta: Union[np.ndarray, str] = None,
                 p: int = 2):
        r"""
        A fourier basis implementation.

        Args:
            input_size (int) : dimensionality of the input
            output_size (int) : dimensionality of the output
            eta (np.ndarray) : Matrix to be used as basis. It must have dimension (out_size x input_size).
                If not provided 0 matrix will be used
        """
        super().__init__(input_size, output_size)

        # out_size x n
        # basis_dimension x state_dimension
        # eta_i = self.eta[i]
        self.eta: np.ndarray = np.zeros((self._output_size, self._input_size)) if eta is None else eta

        if eta == "permutation":
            m = np.zeros(self._output_size * self._input_size)
            p = np.random.permutation(self._output_size * self._input_size)
            ones = int(len(m) / 2)
            m[p[0:ones]] = 1
            self.eta = m.reshape(self._output_size, self._input_size)
        elif eta == "random":
            self.eta = np.random.random((self.output_size(), self.input_size()))
        elif eta is None:
            print("[WARNING] using zero eta matrix")

        assert (self.eta.shape[0] == self._output_size and self.eta.shape[1] == self._input_size)

    def to_basis(self, vector: np.ndarray) -> np.ndarray:
        r"""
        Transform the vector to the Fourier Basis

        Args:
            vector (np.ndarray) : Vector with size (input_size,)
        Returns:
             phi (np.ndarray) : Transformed vector of size (output_size,)
        """
        assert (vector.shape[0] == self._input_size)
        result = np.cos(np.pi * np.dot(self.eta, vector))
        return result

    def save_eta(self, file_name="./eta_fourier_weights.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.eta, f)

    def __call__(self, vector: np.ndarray):
        result = self.to_basis(vector)
        return result

    def scale_learning_rate(self, alpha: float):
        r"""
        Compute learning rate scaled with the norm of elements in the basis

        Args:
            alpha (float) : The learning rate to scale

        Return:
             scaled_lr (np.ndarray) : Vector with learning rate scaled learning rate.
        """

        norm = np.sqrt(np.square(self.eta).sum(axis=1))  # (basis_dimension x 1) or (output_dimension x 1)
        norm[norm == 0] = 1  # When the norm is zero do not scale
        return alpha / norm


class LinearAprox:
    """
    Linearly approximate a function using a basis and weights.
    """

    def __init__(self, basis: Basis, weights: np.ndarray):
        """
        Initialize the linear approximator

        Args:
            basis (Basis) : The basis to use
            weights (np.ndarray) : The weights to use. It must have dimension actions * (hidden). Hidden is the output
                size of the basis
        """
        self.basis = basis
        self.weights = weights  # (a x m)

        # Assert it is possible to do the product between matrices
        assert basis.output_size() == weights.shape[1]

    def __call__(self, state, action):
        """
        Linear approximator of a function
        Args:
            state (np.ndarray) : vector representing the input state. Must have size of the state (state_dim,)
            action(int) : The action taken as index
        Returns:
             approximation (float) : Returns a scalar representing the approximation
        """
        if action >= self.weights.shape[0] or action < 0:
            raise ValueError(f"Action not allowed {action}")

        # Transform the state with the basis
        transformed = self.basis(state)  # m x 1

        # use the weights corresponding to the action
        weights = self.weights[action]  # 1 x m

        # Compute the scalar product
        return np.dot(weights, transformed)


class SarsaLambda:
    r"""
     An agent that learns using Sarsa Lambda. Stochastic Gradient Descent with Nesterov Acceleration is used for training.
    """

    def __init__(self,
                 state_dimension: int,
                 number_actions: int,
                 eta: np.ndarray,
                 discount_factor_gamma: float = 0.99,
                 lambda_sarsa: float = 0.9,
                 momentum: float = 0,
                 exploration_strategy: str = "eps"
                 ):
        r"""
        Initialize the SarsaLambda agent along with its training parameters

        Args:
             state_dimension (int): The dimensionality of the state space
             number_actions (int): The number of actions available on the environment (discrete)
             eta (np.ndarray): The basis to be used in the Fourier Approximation
             discount_factor_gamma (float): The discount factor used in the SARSA update
             lambda_sarsa (float): The lambda used in the SARSA update
             momentum(float): The momentum of the Stochastic Gradient Descent
        """

        # Set the parameters
        self.number_actions: int = number_actions
        self.hidden_size: int = eta.shape[0]
        self.discount_factor_gamma: float = discount_factor_gamma
        self.lambda_sarsa: float = lambda_sarsa
        self.momentum: float = momentum
        self.exploration_strategy: str = exploration_strategy

        # Initialize the weights and the velocity
        self.weights: np.ndarray = np.random.random((number_actions, self.hidden_size))  # n_a x h_s
        self.velocity: np.ndarray = np.zeros((number_actions, self.hidden_size))

        # Create the basis
        self.basis = FourierBasis(input_size=state_dimension, output_size=self.hidden_size, eta=eta)
        self.linear_aprox = LinearAprox(self.basis, self.weights)

        # Initialize eligibility trace
        self.eligibility_trace: np.ndarray = np.zeros((number_actions, self.hidden_size))  # n_a x h_s

    def reset(self) -> None:
        r"""
        Reset the eligibility trace and the velocity.
        """

        self.eligibility_trace = np.zeros((self.number_actions, self.hidden_size))
        self.velocity = np.zeros((self.number_actions, self.hidden_size))

    def explore_greedy(self, state, epsilon=0.1):
        """
        Choose an action using a greedy policy with parameter epsilon.

        Args:
             state (np.ndarray): Vector representing the state
             epsilon (float): Epsilon. Must be a number between 0 and 1.

        Returns:
            best_action (int) : The best action to take with the epsilon greedy policy

        """

        # Bernoulli variable with parameter epsilon
        if np.random.binomial(size=1, n=1, p=epsilon) == 1:

            # Act according to the exploration policy
            if self.exploration_strategy == "opposite":
                best_q = float("inf")
                best_action = 0
                for action in range(self.number_actions):
                    q = self.linear_aprox(state, action)
                    if q < best_q:
                        best_q = q
                        best_action = action

                return best_action
            elif self.exploration_strategy == "state":
                if state[0] < 0.5:
                    return 0
                else:
                    return 2
            elif self.exploration_strategy == "still":
                return 1

            # Take random action
            return np.random.randint(0, self.number_actions)
        else:
            # Compute best action using learnt Q function
            best_q = float("-inf")
            best_action = 0
            for action in range(self.number_actions):
                q = self.linear_aprox(state, action)
                if q > best_q:
                    best_q = q
                    best_action = action

            return best_action

    def forward(self,
                state_t: np.ndarray,
                action_t: int,
                reward_t: float,
                state_t_next: np.ndarray,
                action_t_next: int,
                learning_rate_t: float,
                ) -> None:
        """
        Updates the internal representation of the Q function using Stochastic Gradient Descent with Nesterov Acceleration.

        Internally it updates the eligibility trace , scales the learning rate for each basis and updates weights and velocity
        matrices.

        Args:
            state_t (np.ndarray): Vector representing the state
            action_t (int): Action taken in that state with current policy
            reward_t (float): Reward obtained by taking that action
            state_t_next (np.ndarray): Vector representing the next state
            action_t_next (float): Action to be taken in the enxt state with current policy.
            learning_rate_t (float): The learning rate to apply
        """
        ## UPDATE ELGIBILITY TRACE
        self._update_eligibility_trace(action_t, state_t)

        ## UPDATE WEIGHTS
        delta = self.compute_delta(state_t, state_t_next, action_t, action_t_next, reward_t)

        ## APPLY SGD
        # v <- mv + alpha * delta * e
        # w <- w + v * momentum  + alpha * delta * eligibility
        scaled_learning_rate: np.ndarray = self.basis.scale_learning_rate(learning_rate_t)  # (h_s,)

        for index, lr in enumerate(scaled_learning_rate):
            self.velocity[:, index] = self.velocity[:, index] * self.momentum + \
                                      lr * delta * self.eligibility_trace[:, index]
            self.weights[:, index] = self.weights[:, index] + self.velocity[:, index] * self.momentum
            self.weights[:, index] = self.weights[:, index] + lr * delta * self.eligibility_trace[:, index]

        # todo : understand why matrix implementation does not work
        # repeat
        # scaled_learning_rate = np.repeat(scaled_learning_rate.reshape(1,self.hidden_size),repeats=self.number_actions,axis=0) #(num_action,hidden_size)
        #
        # self.velocity = self.velocity * self.momentum + scaled_learning_rate * delta * self.eligibility_trace  # ()
        # self.weights = self.weights + self.velocity * self.momentum + scaled_learning_rate * delta * self.eligibility_trace

    def _update_eligibility_trace(self, action_t: int, state_t: np.ndarray) -> np.ndarray:
        """
        Update the eligibility trace

        Args:
            action_t (int): Action taken
            state_t (np.ndarray): Vector representing state

        Return:
            eligibility_trace (np.ndarray) : Updated eligibility trace
        """
        transformed_t = self.basis(state_t)
        # Create boolean matrix
        actions = np.zeros((self.number_actions, self.hidden_size))
        actions[action_t, :] = 1

        # Update eligibility trace
        self.eligibility_trace = self.discount_factor_gamma * self.lambda_sarsa * self.eligibility_trace \
                                 + transformed_t * actions

        # clip it for avoiding gradient exploit
        self.eligibility_trace = np.clip(self.eligibility_trace, -5, 5)

        return self.eligibility_trace

    def compute_delta(self,
                      state_t: np.ndarray,
                      state_t_next: np.ndarray,
                      action_t: int,
                      action_t_next: int,
                      reward_t: float) -> float:
        r"""
        Compute the delta

        Args:
            state_t (np.ndarray): Vector representing state_t
            state_t_next(np.ndarray): Vector representing state_t_next
            action_t(int): Action taken in state_t
            action_t_next(int): Action to be taken in state_t_next with current policy
            reward_t(float): Reward obtained after taking action_t in state_t
        Return:
            delta (float) : The delta obtained

        """
        # delta_t = r_t + gamma * Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)

        q_t_next = self.linear_aprox(state_t_next, action_t_next)
        q_t = self.linear_aprox(state_t, action_t)
        return reward_t + self.discount_factor_gamma * q_t_next - q_t

    def plot_value_function(self, file_name: str = "") -> None:
        r"""
        Generate 3D graph of the value function of the model.

        Args:
            file_name (str) : If specified the plot is saved in a file with this path.
        """
        NUMBER_POINTS = 100
        s1 = np.linspace(0, 1, NUMBER_POINTS)
        s2 = np.linspace(0, 1, NUMBER_POINTS)

        # Compute value function in the domain
        value = np.ones((NUMBER_POINTS, NUMBER_POINTS)) * -200

        for i in range(NUMBER_POINTS):
            for j in range(NUMBER_POINTS):
                state = np.array([s1[i], s2[j]])
                phi = self.basis(state)
                q_a = np.dot(self.weights, phi)

                q_max = np.max(q_a)

                value[(i, j)] = q_max

        x, y = np.meshgrid(s1, s2)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('V')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.suptitle('Value Function', fontsize=20)

        if file_name == "":
            plt.show()
        else:
            plt.savefig(file_name)

    def plot_best_action(self, file_name=""):
        NUMBER_POINTS = 100
        s1 = np.linspace(0, 1, NUMBER_POINTS)
        s2 = np.linspace(0, 1, NUMBER_POINTS)

        # COmpute value function
        value = np.ones((NUMBER_POINTS, NUMBER_POINTS)) * -200

        for i in range(NUMBER_POINTS):
            for j in range(NUMBER_POINTS):
                state = np.array([s1[i], s2[j]])
                phi = self.basis(state)
                q_a = np.dot(self.weights, phi)

                action = np.argmax(q_a)

                value[(i, j)] = action - 1

        x, y = np.meshgrid(s1, s2)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Action')

        fig.suptitle('Action', fontsize=20)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if file_name == "":
            plt.show()
            plt.close()
        else:
            plt.savefig(file_name)
            plt.close()

    def save(self, folder=".", file_prefix: str = "", extra_data: Any = None):
        with open(os.path.join(folder, file_prefix + "weights.pkl") + "", "wb") as f:
            content = {"W": self.weights, "N": self.basis.eta, "info": extra_data}
            pickle.dump(content, f)

        with open(os.path.join(folder, file_prefix + "agent.pkl") + "", "wb") as f:
            pickle.dump(self, f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            data: dict = pickle.load(f)
            weights: np.ndarray = data["W"]
            eta: np.ndarray = data["N"]

            self.__init__(state_dimension=eta.shape[1],
                          number_actions=weights.shape[0],
                          eta=eta,
                          )
            self.weights = weights

    def __str__(self):
        return "Sarsa m:{:4.2f} $\lambda$: {:4.2f} $\gamma$: {:4f}".format(self.momentum, self.lambda_sarsa,
                                                                           self.discount_factor_gamma)


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
                 information_episodes: int = 50):

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

    def train(self, verbose=False):
        #np.random.seed(10)
        ### RESET ENVIRONMENT ###
        self.env.reset()

        ### DEFINE FUNCTION USEFUL FOR SCALING ###
        def scale_state_variables(s, low=self.env.observation_space.low, high=self.env.observation_space.high):
            ''' Rescaling of s to the box [0,1]^2 '''
            x = (s - low) / (high - low)
            return x

        epsilon = self._define_epsilon()
        self.learning_rate = self.learning_rate_initial
        time = 0

        for e in trange(self.number_episodes):
            done = False
            terminated = False
            state = scale_state_variables(self.env.reset()[0])
            total_episode_reward = 0.

            while not (done or terminated):
                ### TAKE ACTION ###
                action = self.agent.explore_greedy(state, epsilon=epsilon[e])

                ### USE ACTION ###
                next_state, reward, done, terminated, *_ = self.env.step(action)
                next_state = scale_state_variables(next_state, self.env.observation_space.low,
                                                   self.env.observation_space.high)

                ### COMPUTE NEXT ACTION

                next_action = self.agent.explore_greedy(next_state, epsilon=epsilon[e])

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

            if self._moving_average(30) > self.episode_reward_trigger and self.early_stopping:
                self.env.close()
                break

            self.env.close()
        ### PLOT RESULTS ###

        if verbose:
            self.plot_rewards()

    def test(self, N=50, verbose=False):
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

            action = self.agent.explore_greedy(state, epsilon=0)

            while not (done or truncated):
                # Get next state and reward.  The done variable
                # will be True if you reached the goal position,
                # False otherwise
                next_state, reward, done, truncated, *_ = self.env.step(action)
                next_state = scale_state_variables(next_state)
                next_action = self.agent.explore_greedy(state, epsilon=0)

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
            return True, avg_reward, confidence
        else:
            if verbose:
                print(
                    'Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence'.format(
                        CONFIDENCE_PASS))
            return False, avg_reward, confidence

    def play_game(self):

        def scale_state_variables(s, low=self.env.observation_space.low, high=self.env.observation_space.high):
            ''' Rescaling of s to the box [0,1]^2 '''
            x = (s - low) / (high - low)
            return x

        done = False
        truncated = False
        total_episode_reward = 0

        state = scale_state_variables(self.env.reset()[0])
        action = self.agent.explore_greedy(state, epsilon=0)

        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, *_ = self.env.step(action)
            next_state = scale_state_variables(next_state)
            next_action = self.agent.explore_greedy(state, epsilon=0)

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


def test_basis():
    state = np.ones(2)  # state = [1,1]

    basis = FourierBasis(input_size=state.shape[0], output_size=3)

    result = basis(state)
    print(result)

    custom_eta = np.array([[1, 0],
                           [0, 1],
                           [1, 1]])
    basis = FourierBasis(input_size=state.shape[0], output_size=3, eta=custom_eta)
    result = basis(state)

    # reslut[0] = cos( pi * [1,0] * [1,1]) = cos( pi * 1) = -1
    # result[1] = cos (pi * [0,1] * [1,1]) = cos(pi * 1) = -1
    # result[2] = cost(pi * [1,1] * [1,1]) = cost(pi * 2) = 1
    print(result)


def test_linear():
    state = np.ones(2)  # state = [1,1]
    # state[1] = 0
    custom_eta = np.array([[1, 0],
                           [0, 1],
                           [1, 1]])

    basis = FourierBasis(input_size=state.shape[0], output_size=3, eta=custom_eta)

    action = 1
    weights = np.array([[0.5, 0.5],
                        [1, 1],
                        [0, 1]])

    l_a = LinearAprox(basis, weights.T)
    result = l_a(state, action)
    print(result)


if __name__ == '__main__':
    test_basis()
    test_linear()
