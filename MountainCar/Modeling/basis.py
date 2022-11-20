import numpy as np
import pickle


class FourierBasis:
    def __init__(self, input_dimension, output_dimension, eta: np.ndarray = None, p: int = 2):
        """
        A fourier basis implementation.
        Args:
            input_dimension (int) : dimensionality of the input
            output_dimension (int) : dimensionality of the output
            eta (np.ndarray) : matrix to be used as basis. If not provided 0 matrix will be used
        """
        self.input_dimension = input_dimension
        self.eta: np.ndarray = np.zeros((output_dimension, input_dimension)) if eta is None else eta
        if eta == None:
            print("[WARNING] using zero eta matrix")

        assert (self.eta.shape[0] == output_dimension and self.eta.shape[1] == input_dimension)

    def __call__(self, state: np.ndarray):
        assert (state.shape[0] == self.input_dimension)
        return np.cos(np.pi * np.dot(self.eta, state))

    def save_eta(self, file_name="./eta_fourier_weights.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.eta, f)


def test_basis():
    state = np.ones(2)  # state = [1,1]
    # state[1] = 0

    basis = FourierBasis(input_dimension=state.shape[0], output_dimension=3)

    result = basis(state)
    print(result)

    custom_eta = np.array([[1, 0],
                           [0, 1],
                           [1, 1]])
    basis = FourierBasis(input_dimension=state.shape[0], output_dimension=3, eta=custom_eta)
    result = basis(state)

    # reslut[0] = cos( pi * [1,0] * [1,1]) = cos( pi * 1) = -1
    # result[1] = cos (pi * [0,1] * [1,1]) = cos(pi * 1) = -1
    # result[2] = cost(pi * [1,1] * [1,1]) = cost(pi * 2) = 1
    print(result)


if __name__ == '__main__':
    test_basis()
