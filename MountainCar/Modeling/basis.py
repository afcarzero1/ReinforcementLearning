import os
from abc import ABC, abstractmethod
from typing import Union, Any

import numpy as np
import pickle


class Basis(ABC):
    """
    Generic basis to be applied on vectors.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the basis
        Args:
            input_size(int): Dimension of the starting space
            output_size(int): Dimension of the ending space
        """
        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def to_basis(self, vector: np.ndarray):
        """
        Transform the vector to the given basis
        Args:
            vector (np.ndarray) : vector to transform
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def input_size(self):
        return self._input_size

    def output_size(self):
        return self._output_size


class FourierBasis(Basis):
    def __init__(self, input_size: int, output_size: int, eta: Union[np.ndarray, str] = None,
                 p: int = 2):
        """
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

    def __call__(self, state: np.ndarray):
        return self.to_basis(state)

    def to_basis(self, vector: np.ndarray) -> np.ndarray:
        """
        Transform the vector to the Fourirer Basis
        Args:
            vector (np.ndarray) : Vector with size (input_size,)
        Returns:
             phi (np.ndarray) : Transformed vector of size (output_size,)
        """
        assert (vector.shape[0] == self._input_size)

        return np.cos(np.pi * np.dot(self.eta, vector))

    def save_eta(self, file_name="./eta_fourier_weights.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.eta, f)

    def scale_learning_rate(self, alpha: float):
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
    def __init__(self,
                 state_dimension,
                 number_actions,
                 eta: np.ndarray,
                 discount_factor_gamma: float = 0.99,
                 lambda_sarsa: float = 0.9,
                 momentum: float = 0,
                 ):
        self.number_actions: int = number_actions
        self.hidden_size: int = eta.shape[0]
        self.discount_factor_gamma: float = discount_factor_gamma
        self.lambda_sarsa: float = lambda_sarsa
        self.momentum = momentum
        self.weights: np.ndarray = np.random.random((number_actions, self.hidden_size))  # n_a x h_s
        self.velocity: np.ndarray = np.zeros((number_actions, self.hidden_size))

        self.basis = FourierBasis(input_size=state_dimension, output_size=self.hidden_size, eta=eta)
        self.linear_aprox = LinearAprox(self.basis, self.weights)

        # Initialize eligebility trace
        self.eligibility_trace = np.zeros((number_actions, self.hidden_size))  # n_a x h_s

    def reset(self):
        self.eligibility_trace = np.zeros((self.number_actions, self.hidden_size))
        self.velocity = np.zeros(
            (self.number_actions, self.hidden_size))  # todo : verufy this is correct initialization

    def epsilon_greedy(self, state, epsilon=0.1):
        """
        Choose an action using a greedy policy.
        :param state:
        :param epsilon:
        :return:
        """

        if np.random.binomial(size=1, n=1, p=epsilon) == 1:
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
                ):
        ## UPDATE ELGIBILITY TRACE
        self.update_eligibility_trace(action_t, state_t)

        ## UPDATE WEIGHTS
        delta = self.compute_delta(state_t, state_t_next, action_t, action_t_next, reward_t)

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

    def update_eligibility_trace(self, action_t: int, state_t: np.ndarray):
        transformed_t = self.basis(state_t)
        # Create boolean matrix
        actions = np.zeros((self.number_actions, self.hidden_size))
        actions[action_t, :] = 1

        # Update eligibility trace
        self.eligibility_trace = self.discount_factor_gamma * self.lambda_sarsa * self.eligibility_trace \
                                 + transformed_t * actions

        # clip it
        self.eligibility_trace = np.clip(self.eligibility_trace, -5, 5)

        return self.eligibility_trace

    def compute_delta(self,
                      state_t: np.ndarray,
                      state_t_next: np.ndarray,
                      action_t: int,
                      action_t_next: int,
                      reward_t: float) -> float:
        # delta_t = r_t + gamma * Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)

        q_t_next = self.linear_aprox(state_t_next, action_t_next)
        q_t = self.linear_aprox(state_t, action_t)
        return reward_t + self.discount_factor_gamma * q_t_next - q_t

    def save(self,file_prefix : str = "", extra_data : Any = None):
        with open(os.path.join(".",file_prefix+"_weights.pkl") + "","wb") as f:
            content = {"W" : self.weights , "N" :self.basis.eta , "info" : extra_data}
            pickle.dump(content,f)

    def __str__(self):
        return "Sarsa m:{:4.2f} $\lambda$: {:4.2f} $\gamma$: {:4f}".format(self.momentum,self.lambda_sarsa,self.discount_factor_gamma)

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
