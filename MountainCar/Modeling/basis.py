from abc import ABC, abstractmethod

import numpy as np
import pickle


class Basis(ABC):
    def __init__(self, input_size: int, output_size: int):
        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def to_basis(self, vector: np.ndarray):
        """
        Transform the vector to the given basis
        Args:
            vector (np.ndarray) : vector to trandform
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
    def __init__(self, basis: Basis, weights: np.ndarray):
        self.basis = basis
        self.weights = weights  # (a x m)

        # Assert it is possible to do the product
        assert basis.output_size() == weights.shape[1]

    def __call__(self, state, action):
        if action >= self.weights.shape[0] or action < 0:
            raise ValueError(f"Action not allowed {action}")

        # Transform the state with the basis
        transformed = self.basis(state)  # m x 1

        # use the weights
        weights = self.weights[action]  # 1xm
        return np.dot(weights, transformed)


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
    weights = np.array([[0.5,0.5],
                        [1,1],
                        [0,1]])

    l_a = LinearAprox(basis , weights.T )
    result = l_a(state,action)
    print(result)



if __name__ == '__main__':
    test_basis()
    test_linear()
