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
                 orthogonal_init=False, adaptive=False, linear=False, learn_inv=False):
        # Storing layer parameters
        self.transfer_func = transfer_func
        self.transfer_derivative_func = transfer_derivative_func
        self.transfer_inverse_func = transfer_inverse_func
        self.linear = linear # Note that this will mean no transfer functions are used

        # Generating an (initial gaussian random) weight matrix, inv matrix and biases
        if seed is not None:
            xp.random.seed(seed)

        # Initialize weights
        self.forward_weight_matrix = xp.random.randn(nb_units, nb_units)
        self.forward_weight_matrix *= xp.sqrt(2/(self.forward_weight_matrix.shape[0]+self.forward_weight_matrix.shape[1]))
        self.forward_biases = xp.zeros(nb_units)
        
        self.backward_weight_matrix = xp.random.randn(nb_units, nb_units)
        self.backward_weight_matrix *= xp.sqrt(2/(self.backward_weight_matrix.shape[0]+self.backward_weight_matrix.shape[1]))
        self.backward_biases = xp.zeros(nb_units)

        # If we want an orthogonal init, we have to correct the shape
        if orthogonal_init:
            np.random.seed(seed)
            self.forward_weight_matrix = xp.asarray(ortho_group.rvs(nb_units))
            self.backward_weight_matrix = xp.asarray(ortho_group.rvs(nb_units))

        if not learn_inv:
            self.update_exact_inverse() # Inverse defined here
    
        self.first_moment_factor = 0.9
        self.second_moment_factor = 0.99

        # If we want an adaptive learning rate, we need parameter-wise multipliers
        # This is initially adagrad
        self.adaptive = adaptive
        if self.adaptive:
            self.adaptive_forward_weight_params = {'first_moment': xp.zeros(self.forward_weight_matrix.shape), 'second_moment': xp.zeros(forward_weight_matrix.shape)}
            self.adaptive_forward_bias_params = {'first_moment': xp.zeros(self.forward_biases.shape), 'second_moment': xp.zeros(forward_biases.shape)}
            if learn_inv:
                self.adaptive_backward_weight_params = {'first_moment': xp.zeros(self.backward_weight_matrix.shape), 'second_moment': xp.zeros(backward_weight_matrix.shape)}
                self.adaptive_backward_bias_params = {'first_moment': xp.zeros(self.backward_biases.shape), 'second_moment': xp.zeros(backward_biases.shape)}


    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        weight_transform = xp.einsum('ij, ni -> nj', self.forward_weight_matrix, data) + self.forward_biases
        if self.linear:
            return weight_transform
        return self.transfer_func(weight_transform)

    def inverse(self, data):
        if self.linear:
            return xp.einsum('ji, nj -> ni', self.backward_weight_matrix, data - self.backward_biases)
        return xp.einsum('ji, nj -> ni', self.backward_weight_matrix, self.transfer_inverse_func(data) - self.backward_biases)

    def update_exact_inverse(self):
        self.backward_weight_matrix = xp.linalg.inv(self.forward_weight_matrix)
        self.backward_biases = xp.copy(self.forward_biases)

    def adaptive_update(self, direction, learning_rate, param_dict):
        param_dict['first_moment'] = (1.0 - self.first_moment_factor)*direction + self.first_moment_factor*self.param_dict['first_moment']
        param_dict['second_moment'] = (1.0 - self.second_moment_factor)*direction**2 + self.second_moment_factor*param_dict['second_moment']

        adjusted_learning_rate = (learning_rate / (1e-8 + xp.sqrt(param_dict['second_moment'])))
        return adjusted_learning_rate*self.param_dict['first_moment']

    def update_forward_weights(self, direction, learning_rate):
        if self.adaptive:
            self.forward_weight_matrix += self.adaptive_update(direction, learning_rate, self.adaptive_forward_weight_params)
        else:
            self.forward_weight_matrix += learning_rate*direction

    def update_forward_biases(self, direction, learning_rate):
        if self.adaptive:
            self.forward_biases += self.adaptive_update(direction, learning_rate, self.adaptive_forward_bias_params)
        else:
            self.forward_biases += learning_rate*direction

    def update_backward_weights(self, direction, learning_rate):
        if self.adaptive:
            self.backward_weight_matrix += self.adaptive_update(direction, learning_rate, self.adaptive_backward_weight_params)
        else:
            self.backward_weight_matrix += learning_rate*direction

    def update_backward_biases(self, direction, learning_rate):
        if self.adaptive:
            self.backward_biases += self.adaptive_update(direction, learning_rate, self.adaptive_backward_bias_params)
        else:
            self.backward_biases += learning_rate*direction

    def save_params(self, path):
        xp.save(path + "forward_weights.npy", self.forward_weight_matrix)
        xp.save(path + "forward_biases.npy", self.forward_biases)
        xp.save(path + "backward_weights.npy", self.backward_weight_matrix)
        xp.save(path + "backward_biases.npy", self.backward_biases)

    def load_params(self, path):
        self.forward_weight_matrix = xp.load(path + "forward_weights.npy")
        self.forward_biases = xp.load(path + "forward_biases.npy")
        self.backward_weight_matrix = xp.load(path + "backward_weights.npy")
        self.backward_biases = xp.load(path + "backward_biases.npy")
