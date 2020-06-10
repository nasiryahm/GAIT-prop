is_cupy = True
try:
    import cupy as xp
except ImportError:
    print("Unable to load cupy. Falling back to numpy.")
    import numpy as xp
    is_cupy = False

import numpy as np

# leaky ReLu functions
def leaky_relu(data):
    out = xp.copy(data)
    out[out < 0.0] *= 0.1
    return out

def leaky_relu_derivative(data):
    out = xp.empty(data.shape)
    out[data >= 0.0] = 1.0
    out[data < 0.0] = 0.1
    return out

def leaky_relu_inverse(data):
    out = xp.copy(data)
    out[out < 0.0] *= (1.0 / 0.1)
    return out


def indices_to_onehot(data, nb_categories=10):
    # Such indexing as done below only works with numpy arrays
    onehot = np.zeros((len(data), nb_categories))
    onehot[range(len(data)), data] = 1.0
    return onehot

def performance(net, input_data, target_data):
    output = net.forward_pass(input_data)
    if is_cupy:
        correct_mask = xp.asnumpy(xp.argmax(output[-1][:,:10], axis=1)) == xp.asnumpy(xp.argmax(target_data, axis=1))
        loss = float(xp.sum(xp.asnumpy((xp.mean(output[-1][:,:10]) - target_data) ** 2)))
    else:
        correct_mask = xp.argmax(output[-1][:,:10], axis=1) == xp.argmax(target_data, axis=1)
        loss = float(xp.sum(xp.mean((output[-1][:,:10]) - target_data) ** 2))
    accuracy = (np.sum(correct_mask) / np.size(correct_mask))
    # Assuming MSE loss
    return accuracy, loss

def progress(x, y):
    out = '%s Percent Complete: %s acc (training)' % (x, y)  # The output
    bs = '\r'            # The backspace
    print(bs, end="")
    print(out, end="", flush=True)

