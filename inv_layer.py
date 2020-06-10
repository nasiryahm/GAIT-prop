from scipy.stats import ortho_group
try:
    import cupy as xp
except ImportError:
    print("Cupy couldn't be imported. Falling back to numpy.")
    import numpy as xp
import numpy as np



class InvLayer:
    """ Invertible NN Layer"""

    def __init__(self, nb_units, seed=1,
                 transfer_func=lambda x: x, transfer_derivative_func=lambda x: 1, transfer_inverse_func=lambda x: x,
                 orthogonal_init=False, adaptive=False, linear=False):
        # Storing layer parameters
        self.transfer_func = transfer_func
        self.transfer_derivative_func = transfer_derivative_func
        self.transfer_inverse_func = transfer_inverse_func
        self.linear = linear # Note that this will mean no transfer functions are used

        # Generating an (initial gaussian random) weight matrix, inv matrix and biases
        if seed is not None:
            xp.random.seed(seed)

        # Initialize weights
        self.weight_matrix = xp.random.randn(nb_units, nb_units)
        # Applying a scaling to the weight matrix (Xavier init)
        self.weight_matrix *= xp.sqrt(2/(self.weight_matrix.shape[0]+self.weight_matrix.shape[1]))

        # If we want an orthogonal init, we have to correct the shape
        if orthogonal_init:
            np.random.seed(seed)
            full_weights = xp.asarray(ortho_group.rvs(nb_units))
            self.weight_matrix = full_weights

        self.inverse_matrix = None
        self.update_inverse() # Inverse defined here
        self.biases = xp.zeros(nb_units)
    
        self.first_moment_factor = 0.9
        self.second_moment_factor = 0.99

        # If we want an adaptive learning rate, we need parameter-wise multipliers
        # This is initially adagrad
        self.adaptive = adaptive
        if self.adaptive:
            self.first_moment_factor_weights = xp.zeros(self.weight_matrix.shape)
            self.first_moment_factor_biases = xp.zeros(self.biases.shape)
            self.second_moment_factor_weights = xp.zeros(self.weight_matrix.shape)
            self.second_moment_factor_biases = xp.zeros(self.biases.shape)

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        weight_transform = xp.einsum('ij, ni -> nj', self.weight_matrix, data) + self.biases
        if self.linear:
            return weight_transform
        return self.transfer_func(weight_transform)

    def inverse(self, data):
        if self.linear:
            return xp.einsum('ji, nj -> ni', self.inverse_matrix, data - self.biases)
        return xp.einsum('ji, nj -> ni', self.inverse_matrix, self.transfer_inverse_func(data) - self.biases)

    def update_inverse(self):
        self.inverse_matrix = xp.linalg.inv(self.weight_matrix)

    def update_weights(self, direction, learning_rate):
        if self.adaptive:
            self.first_moment_factor_weights = (1.0 - self.first_moment_factor)*direction + self.first_moment_factor*self.first_moment_factor_weights
            self.second_moment_factor_weights = (1.0 - self.second_moment_factor)*direction**2 + self.second_moment_factor*self.second_moment_factor_weights

            learning_rate = (learning_rate / (1e-8 + xp.sqrt(self.second_moment_factor_weights)))

            self.weight_matrix += learning_rate*self.first_moment_factor_weights
        else:
            self.weight_matrix += learning_rate*direction

    def update_biases(self, direction, learning_rate):
        if self.adaptive:
            self.first_moment_factor_biases = (1.0 - self.first_moment_factor)*direction + self.first_moment_factor*self.first_moment_factor_biases
            self.second_moment_factor_biases = (1.0 - self.second_moment_factor)*direction**2 + self.second_moment_factor*self.second_moment_factor_biases

            learning_rate = (learning_rate / (1e-8 + xp.sqrt(self.second_moment_factor_biases)))

            self.biases += learning_rate*self.first_moment_factor_biases
        else:
            self.biases += learning_rate*direction

    def save_params(self, path):
        xp.save(path + "weights.npy", self.weight_matrix)
        xp.save(path + "biases.npy", self.biases)

    def load_params(self, path):
        self.weight_matrix = xp.load(path + "weights.npy")
        self.biases = xp.load(path + "biases.npy")
