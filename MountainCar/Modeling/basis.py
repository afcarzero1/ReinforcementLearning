from abc import ABC, abstractmethod

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
    def __init__(self, input_size: int, output_size: int, eta: np.ndarray = None,
                 p: int = 2):
        """
        A fourier basis implementation.
        Args:
            input_size (int) : dimensionality of the input
            output_size (int) : dimensionality of the output
            eta (np.ndarray) : matrix to be used as basis. If not provided 0 matrix will be used
        """
        super().__init__(input_size, output_size)

        # out_size x n
        # eta_i = self.eta[i]
        self.eta: np.ndarray = np.zeros((self._output_size, self._input_size)) if eta is None else eta
        if eta is None:
            print("[WARNING] using zero eta matrix")

        assert (self.eta.shape[0] == self._output_size and self.eta.shape[1] == self._input_size)

    def __call__(self, state: np.ndarray):
        return self.to_basis(state)

    def to_basis(self, vector: np.ndarray):
        assert (vector.shape[0] == self._input_size)
        return np.cos(np.pi * np.dot(self.eta, vector))

    def save_eta(self, file_name="./eta_fourier_weights.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.eta, f)

    def scale_learning_rate(self, alpha : float):
        norm = np.sqrt(np.square(self.eta).sum(axis=1)) # ax1
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
        if action >= self.weights.shape[0] or action < 0:
            raise ValueError(f"Action not allowed {action}")

        # Transform the state with the basis
        transformed = self.basis(state)  # m x 1

        # use the weights
        weights = self.weights[action]  # 1xm
        return np.dot(weights, transformed)

    def get_scaled_learning_rate(self):
        pass

class SarsaLambda:
    def __init__(self,
                 number_states,
                 number_actions,
                 hidden_size: int = 10,
                 discount_facotor_gamma: float = 0.99,
                 lambda_sarsa: float = 0.9,
                 momentum : float = 0):
        self.number_actions: int = number_actions
        self.hidden_size: int = hidden_size
        self.discount_factor_gamma: float = discount_facotor_gamma
        self.lambda_sarsa: float = lambda_sarsa
        self.momentum = momentum
        self.weights: np.ndarray = np.random.random((number_actions, hidden_size))  # n_a x h_s
        self.velocity : np.ndarray = np.random.random((number_actions, hidden_size))

        self.basis = FourierBasis(input_size=number_states, output_size=hidden_size)
        self.linear_aprox = LinearAprox(self.basis, self.weights)

        # Initialize eligebility trace
        self.eligibility_trace = np.zeros((number_actions, hidden_size))  # n_a x h_s

        # clip it
        np.clip(self.eligibility_trace,-5,5)

    def reset(self):
        self.eligibility_trace = np.zeros((self.number_actions, self.hidden_size))

    def forward(self,
                state_t: np.ndarray,
                action_t: int,
                reward_t: float,
                state_t_next: np.ndarray,
                action_t_next: int,
                learning_rate_t: float,
                ):
        ## UPDATE ELGIBILITY TRACE
        self.update_elegibility_trace(action_t, state_t)

        ## UPDATE WEIGHTS
        delta = self.compute_delta(state_t,state_t_next,action_t,action_t_next,reward_t)

        # v <- mv + alpha * delta * e
        # w <- w + valpha * delta * eligibility
        # todo : use here self.base.scale_learning_rate to scale
        self.velocity = self.velocity * self.momentum + learning_rate_t * delta * self.eligibility_trace
        self.weights = self.weights + self.velocity

    def update_elegibility_trace(self, action_t: int, state_t : np.ndarray):
        transformed_t = self.basis(state_t)
        # Create boolean vector
        actions = np.zeros(self.number_actions)
        actions[action_t] = 1

        # Update eligibility trace
        self.eligibility_trace = self.discount_factor_gamma * self.lambda_sarsa * self.eligibility_trace \
                                 + transformed_t * actions

    def compute_delta(self,
                      state_t: np.ndarray,
                      state_t_next: np.ndarray,
                      action_t : int,
                      action_t_next : int,
                      reward_t: float) -> float:
        # delta_t = r_t + gamma * Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)

        q_t_next = self.linear_aprox(state_t_next,action_t_next)
        q_t = self.linear_aprox(state_t,action_t)
        return reward_t + self.discount_factor_gamma * q_t_next - q_t


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
