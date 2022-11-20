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


class SarsaLambda:
    def __init__(self,
                 number_states,
                 number_actions,
                 hidden_size: int = 10,
                 discount_facotor_gamma: float = 0.99,
                 lambda_sarsa: float = 0.9):
        self.number_actions = number_actions
        self.hidden_size = hidden_size
        self.discount_factor_gamma = discount_facotor_gamma
        self.lambda_sarsa = lambda_sarsa
        self.weights = np.random.random((number_actions, hidden_size)) # n_a x h_s

        self.basis = FourierBasis(input_size=number_states, output_size=hidden_size)
        self.linear_aprox = LinearAprox(self.basis, self.weights)

        # Initialize eligebility trace
        self.eligibility_trace = np.zeros((number_actions, hidden_size)) # n_a x h_s

    def reset(self):
        self.eligibility_trace = np.zeros((self.number_actions, self.hidden_size))

    def forward(self,
                state_t : np.ndarray,
                action_t : int,
                reward_t : float,
                state_t_next : np.ndarray,
                action_t_next : int,
                learning_rate_t : float):
        # Compute the states in the basis. Using the approximation
        transformed_t = self.basis(state_t)  # this is also gradient of elegebility trace wrt w_t
        transformed_t_next = self.basis(state_t_next)

        ## UPDATE ELGIBILITY TRACE
        self.update_elegibility_trace(action_t,transformed_t)

        ## UPDATE WEIGHTS
        delta = self.compute_delta(transformed_t,transformed_t_next,reward_t)
        # w <- w + alpha * delta * eligibility
        self.weights = self.weights + learning_rate_t * delta * self.eligibility_trace

    def update_elegibility_trace(self, action_t: int, gradient):

        # Create boolean vector
        actions = np.zeros(self.number_actions)
        actions[action_t] = 1

        # Update eligibility trace
        self.eligibility_trace = self.discount_factor_gamma * self.lambda_sarsa * self.eligibility_trace \
                                 + gradient * actions

    def compute_delta(self,
                      transformed_t : np.ndarray,
                      transformed_t_next : np.ndarray,
                      reward_t : float) -> float:

        # delta_t = r_t + gamma * Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)
        return reward_t + self.discount_factor_gamma * transformed_t_next - transformed_t

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
