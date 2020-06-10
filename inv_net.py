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
        for layer in self.layers:
            output.append(layer.forward(output[-1]))
        return output

    def inverse_pass(self, data):
        output = [data]
        # For the inverse direction, we invert layer order
        for layer in self.layers[::-1]:
            output.append(layer.inverse(output[-1]))
        # Finally, make the output as we would see the forward pass
        return output[::-1]

    def update_inverse_model(self):
        for layer in self.layers[1:]:
            layer.update_inverse()

    def save_params(self, out_folder):
        # Layer-wise dump weights to the output folder
        for index, layer in enumerate(self.layers):
            layer.save_params(out_folder + str(index) + "L")

    def load_params(self, out_folder):
        # Layer-wise load weights
        for index, layer in enumerate(self.layers):
            layer.load_params(out_folder + str(index) + "L")

    def ortho_gradients(self, ortho_weighting, layer_index):
        weights = Variable(self.layers[layer_index].weight_matrix)
        reg = functions.einsum('ik, jk -> ij', weights, weights)

        target = reg * xp.eye(self.layers[layer_index].weight_matrix.shape[0])
        ortho_loss = functions.sum((reg - target) ** 2)
        gradient = grad([ortho_loss], [weights])[0].array
        return ortho_weighting * gradient

    def update_parameters(self, weight_updates, bias_updates, learning_rate):
        for layer_index in range(len(self.layers)):
            self.layers[layer_index].update_weights(weight_updates[layer_index], learning_rate)
            self.layers[layer_index].update_biases(bias_updates[layer_index], learning_rate)

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
            error = mult_factor * (forward_pass[layer_index + 1] - inverse)

            if self.layers[layer_index].linear:
                layer_derivatives = xp.ones((error.shape))
            else:
                layer_derivatives = self.layers[layer_index].transfer_derivative_func(
                    self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index + 1]))
                error *= layer_derivatives

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', error, forward_pass[layer_index]),
                                    axis=0)
            bias_update = xp.mean(error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, layer_index)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

            # Adjust and calculate the next layers target
            grad_adjusted_inc_factor = gamma * layer_derivatives * layer_derivatives
            inverse = self.layers[layer_index].inverse(
                ((1.0 - grad_adjusted_inc_factor) * forward_pass[layer_index + 1] + grad_adjusted_inc_factor * inverse))
            mult_factor = mult_factor / gamma

        return weight_updates[::-1], bias_updates[::-1]

    def get_tp_updates(self, forward_pass, targets, ortho_weighting=0.0):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        # We must compute layer-wise targets by direct inverse
        inverse_pass = self.inverse_pass(targets)

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
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', error, forward_pass[layer_index]), axis=0)
            bias_update = xp.mean(error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, layer_index)

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
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', error, forward_pass[layer_index]),
                                    axis=0)
            bias_update = xp.mean(error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, layer_index)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

            # Propagate the error to the next layer
            error = xp.einsum('nj, ij -> ni', error, self.layers[layer_index].weight_matrix)

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

    def forward_pass(self, data):
        output = [data]
        for indx, layer in enumerate(self.layers):
            output.append(layer.forward(output[-1][:, :self.net_structure[indx + 1]]))
        return output

    # This inverse_pass has different arguments to SquareInvNet since aux neuron activities from forward are needed
    def inverse_pass(self, data, forward):
        output = [data]
        # For the inverse direction, we invert layer order
        for indx, layer in enumerate(self.layers[::-1]):
            out = layer.inverse(output[-1])
            output.append(xp.hstack([out, forward[-2 - indx][:, out.shape[1]:]]))
        return output[::-1]

    def get_gait_updates(self, forward_pass, targets, ortho_weighting=0.0, gamma=0.001):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        nb_layers = len(self.layers)

        inverse = targets
        mult_factor = 1.0

        for layer_index in range(nb_layers)[::-1]:
            error = mult_factor * (forward_pass[layer_index + 1] - inverse)

            if self.layers[layer_index].linear:
                layer_derivatives = xp.ones((error.shape))
            else:
                layer_derivatives = self.layers[layer_index].transfer_derivative_func(
                    self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index + 1]))

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', layer_derivatives * error,
                                              forward_pass[layer_index][:, :self.net_structure[layer_index + 1]]),
                                    axis=0)
            bias_update = xp.mean(layer_derivatives * error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, layer_index)

            # Collect updates
            weight_updates.append(-weight_update)
            bias_updates.append(-bias_update)

            grad_adjusted_inc_factor = gamma * layer_derivatives * layer_derivatives
            inverse = self.layers[layer_index].inverse(
                ((1.0 - grad_adjusted_inc_factor) * forward_pass[layer_index + 1] + grad_adjusted_inc_factor * inverse))
            mult_factor = mult_factor / gamma

            # Adding the auxilliary neurons on
            inverse = xp.hstack([inverse, forward_pass[layer_index][:, self.net_structure[layer_index + 1]:]])
        return weight_updates[::-1], bias_updates[::-1]

    def get_tp_updates(self, forward_pass, targets, ortho_weighting=0.0):
        # Updates will be stored and returned
        weight_updates = []
        bias_updates = []

        # Forward pass activity is necessary for aux neuron appending
        inverse_pass = self.inverse_pass(targets, forward_pass)

        nb_layers = len(self.layers)
        for layer_index in range(nb_layers)[::-1]:
            # Calculating weight updates based upon the improved target-prop
            error = (forward_pass[layer_index + 1] - inverse_pass[layer_index + 1])

            if self.layers[layer_index].linear:
                layer_derivatives = xp.ones((error.shape))
            else:
                layer_derivatives = self.layers[layer_index].transfer_derivative_func(
                    self.layers[layer_index].transfer_inverse_func(forward_pass[layer_index + 1]))

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', layer_derivatives * error,
                                              forward_pass[layer_index][:, :self.net_structure[layer_index + 1]]),
                                    axis=0)
            bias_update = xp.mean(layer_derivatives * error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, layer_index)

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

            # Calculate updates for this layer
            weight_update = xp.mean(xp.einsum('nj, ni -> nij', layer_derivatives * error,
                                              forward_pass[layer_index][:, :self.net_structure[layer_index + 1]]),
                                    axis=0)
            bias_update = xp.mean(layer_derivatives * error, axis=0)

            # Calculating a weight update based upon a soft orthogonal regularizer
            if ortho_weighting != 0.0:
                weight_update += self.ortho_gradients(ortho_weighting, layer_index)

            # Collect updates
            weight_updates.append(- weight_update)
            bias_updates.append(- bias_update)

            # Propagate the error to the next layer
            error *= layer_derivatives
            error = xp.einsum('nj, ij -> ni', error, self.layers[layer_index].weight_matrix)
            error = xp.hstack([error, xp.zeros((error.shape[0], self.net_structure[layer_index] - error.shape[1]))])

        return weight_updates[::-1], bias_updates[::-1]
