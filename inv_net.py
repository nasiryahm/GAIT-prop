from inv_layer import InvLayer
from chainer import Variable, grad
from chainer import functions

try:
    import cupy as xp
except ImportError:
    print("Cupy couldn't be imported. Falling back to numpy.")
    import numpy as xp


class SquareInvNet:
    def __init__(self, nb_units, nb_hidden_layers, start_seed=None, **kwargs):
        # Construct the invertible layers
        self.net_structure = [nb_units for x in range(nb_hidden_layers + 2)]
        self.layers = []
        this_seed = None
        for n in range(nb_hidden_layers):
            if start_seed is not None:
                this_seed = start_seed + n
            self.layers.append(InvLayer(nb_units, seed=this_seed, **kwargs))
        # Hidden -> Output layer
        self.layers.append(InvLayer(nb_units, seed=this_seed, **kwargs))

    def forward_pass(self, data):
        output = [data]
        for indx, layer in enumerate(self.layers):
            output.append(layer.forward(output[-1][:, :self.net_structure[indx + 1]]))
        return output
    
    def inverse_pass(self, data, forward):
        output = [data]
        # For the inverse direction, we invert layer order
        for indx, layer in enumerate(self.layers[::-1]):
            out = layer.backward(output[-1])
            output.append(xp.hstack([out, forward[-2 - indx][:, out.shape[1]:]]))
        # Finally, make the output as we would see the forward pass
        return output[::-1]

    def create_exact_inverse_model(self):
        for layer in self.layers[1:]:
            layer.update_exact_inverse()

    def save_params(self, out_folder):
        # Layer-wise dump weights to the output folder
        for index, layer in enumerate(self.layers):
            layer.save_params(out_folder + str(index) + "L")

    def load_params(self, out_folder):
        # Layer-wise load weights
        for index, layer in enumerate(self.layers):
            layer.load_params(out_folder + str(index) + "L")

    def ortho_gradients(self, ortho_weighting, weight_matrix):
        weights = Variable(weight_matrix)
        reg = functions.einsum('ik, jk -> ij', weights, weights)

        target = reg * xp.eye(weight_matrix.shape[0])
        ortho_loss = functions.sum((reg - target) ** 2)
        gradient = grad([ortho_loss], [weights])[0].array
        return ortho_weighting * gradient

    def update_forward_parameters(self, weight_updates, bias_updates, learning_rate):
        for layer_index in range(len(self.layers)):
            self.layers[layer_index].update_forward_weights(weight_updates[layer_index], learning_rate)
            self.layers[layer_index].update_forward_biases(bias_updates[layer_index], learning_rate)

    def update_backward_parameters(self, weight_updates, bias_updates, learning_rate):
        for layer_index in range(len(self.layers)):
            self.layers[layer_index].update_backward_weights(weight_updates[layer_index], learning_rate)
            self.layers[layer_index].update_backward_biases(bias_updates[layer_index], learning_rate)

    def get_gait_updates(self, forward_pass, targets, ortho_weighting=0.0, gamma=0.001):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        # We must compute errors layer-wise.
        # In our formulation, each layer's error is partly difference
        nb_layers = len(self.layers)

        # Calculating the inverse target
        inverse = targets
        mult_factor = 1.0

        # Running backwards through layers
        for layer_index in range(nb_layers)[::-1]:

            if self.layers[layer_index].linear:
                layer_derivatives = xp.ones((error.shape))
            else:
                layer_derivatives = self.layers[layer_index].transfer_derivative_func(
                    self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index + 1]))

            grad_adjusted_inc_factor = gamma * layer_derivatives * layer_derivatives
            mult_factor = mult_factor / gamma
            target = (1.0 - grad_adjusted_inc_factor) * forward_pass[layer_index + 1] + grad_adjusted_inc_factor * inverse
            
            error = mult_factor * (forward_pass[layer_index + 1] - target)
            error /= layer_derivatives

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij',error,
                                              forward_pass[layer_index][:, :self.net_structure[layer_index + 1]]),
                                    axis=0)
            bias_update = xp.mean(error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, self.layers[layer_index].forward_weight_matrix)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

            # Adjust and calculate the next layers target
            inverse = self.layers[layer_index].backward(target)
            
            # Add the auxilliary neurons on
            inverse = xp.hstack([inverse, forward_pass[layer_index][:, self.net_structure[layer_index + 1]:]])

        return weight_updates[::-1], bias_updates[::-1]

    def get_tp_updates(self, forward_pass, targets, ortho_weighting=0.0):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        # We must compute layer-wise targets by direct inverse
        inverse_pass = self.inverse_pass(targets, forward_pass)

        nb_layers = len(self.layers)

        for layer_index in range(nb_layers)[::-1]:
            # layer-wise error
            error = (forward_pass[layer_index + 1] - inverse_pass[layer_index + 1])

            if self.layers[layer_index].linear:
                layer_derivatives = xp.ones((error.shape))
            else:
                layer_derivatives = self.layers[layer_index].transfer_derivative_func(
                    self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index + 1]))
                error *= layer_derivatives

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', error,
                                              forward_pass[layer_index][:, :self.net_structure[layer_index + 1]]),
                                    axis=0)
            bias_update = xp.mean(error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, self.layers[layer_index].forward_weight_matrix)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

        return weight_updates[::-1], bias_updates[::-1]

    def get_backprop_updates(self, forward_pass, target, ortho_weighting=0.0):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        # The update will be done layer-wise with a backpropagating signal
        nb_layers = len(self.layers)
        error = forward_pass[-1] - target
        for layer_index in range(nb_layers)[::-1]:
            if self.layers[layer_index].linear:
                layer_derivatives = xp.ones((error.shape))
            else:
                layer_derivatives = self.layers[layer_index].transfer_derivative_func(
                    self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index + 1]))
                error *= layer_derivatives

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', error,
                                              forward_pass[layer_index][:, :self.net_structure[layer_index + 1]]),
                                    axis=0)
            bias_update = xp.mean(error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, self.layers[layer_index].forward_weight_matrix)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

            # Propagate the error to the next layer
            error = xp.einsum('nj, ij -> ni', error, self.layers[layer_index].forward_weight_matrix)
            error = xp.hstack([error, xp.zeros((error.shape[0], self.net_structure[layer_index] - error.shape[1]))])

        return weight_updates[::-1], bias_updates[::-1]

    def get_backward_updates(self, forward_pass):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        # We must compute errors layer-wise.
        # In our formulation, each layer's error is partly difference
        nb_layers = len(self.layers)

        # Running backwards through layers
        for layer_index in range(nb_layers)[::-1]:
            error = self.layers[layer_index].backward(forward_pass[layer_index+1]) - forward_pass[layer_index]

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', error,
                                              self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index+1]) - self.layers[layer_index].backward_biases),
                                    axis=0)
            bias_update = -xp.mean(xp.einsum('nj, ji -> ni', error, self.layers[layer_index].backward_weight_matrix), axis=0)

            ## Calculating a weight update based upon a soft orthogonal regularizer
            #if ortho_weighting != 0.0:
            #    weight_update += self.ortho_gradients(ortho_weighting, self.layers[layer_index].forward_weight_matrix)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

        return weight_updates[::-1], bias_updates[::-1]


class TowerInvNet(SquareInvNet):
    def __init__(self, net_structure=[], start_seed=None, **kwargs):
        # Construct the invertible layers
        self.net_structure = net_structure
        self.layers = []
        this_seed = None
        for n in range(len(net_structure) - 1):
            if start_seed is not None:
                this_seed = start_seed + n
            self.layers.append(InvLayer(net_structure[n + 1], seed=this_seed, **kwargs))
