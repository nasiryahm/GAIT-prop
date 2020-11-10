from inv_net import SquareInvNet
from inv_net import TowerInvNet
from keras.datasets import mnist, fashion_mnist, cifar10
import os
import getopt
import sys
import time
from utils import *
import numpy as np

is_cupy = True
try:
    import cupy as xp
except ImportError:
    print("Cupy couldn't be imported. Falling back to numpy.")
    import numpy as xp
    is_cupy = False



# Default parameters for the set of simulation options
device = 0
nb_epochs = 100
nb_layers = 1
algorithm = "BP"
ortho_init = False
ortho_reg = 0.0
load = False
dataset = 'MNIST'
seed = 1
learning_rate = 0.0001
linear = False
tower = False
learn_inv = False

# Setting up the options for simulation
opts, remaining = getopt.getopt(
    sys.argv[1:],
    '',
    ['algorithm=',
     'load',
     'linear',
     'tower',
     'orthogonal_init',
     'learn_inv',
     'seed=',
     'orthogonal_reg=',
     'device=',
     'nb_epochs=',
     'nb_layers=',
     'learning_rate=',
     'dataset='])
for opt, arg in opts:
    if opt == '--algorithm':
        algorithm = arg
    if opt == '--orthogonal_init':
        ortho_init = True
    if opt == '--learn_inv':
        learn_inv = True
    if opt == '--local_connectivity':
        local_connectivity = True
    if opt == '--load':
        load = True
    if opt == '--linear':
        linear = True
    if opt == '--seed':
        seed = int(arg)
    if opt == '--orthogonal_reg':
        ortho_reg = float(arg)
    if opt == '--nb_layers':
        nb_layers = int(arg)
    if opt == '--device':
        device = int(arg)
    if opt == '--nb_epochs':
        nb_epochs = int(arg)
    if opt == '--dataset':
        dataset = arg
    if opt == '--learning_rate':
        learning_rate = float(arg)
    if opt == '--tower':
        tower = True

if is_cupy:
    xp.cuda.Device(device).use()

assert (dataset in ['MNIST', 'FMNIST', 'KMNIST']), "Dataset is not compatible. Choose MNIST/FMNIST/KMNIST."
assert (algorithm in ['BP', 'GAIT', 'TP']), "Choose either BP, GAIT, or TP traning."

# Prepping the output folder (CUSTOM NAMING FOR SETTINGS)
outpath = "./results/"
outpath += dataset + "_"
print("Using Algorithm: " + algorithm)
outpath = outpath + algorithm + "_"

if linear:
    outpath = outpath + "Linear_"

if orthogonal_init::
    outpath = outpath + "OrthoInit_"

outpath = outpath + str(ortho_reg) + "OrthoReg_"
# If we are using a non-zero orthogonal regularizer, initialise orthogonal matrices (for SquareInvNet).
if ortho_reg == 0.0:
    ortho_init = True
outpath = outpath + str(seed) + "Seed_"
outpath = outpath + str(learning_rate) + "LR_"

if tower:
    outpath = outpath + str(nb_layers) + "TOWER/"
else:
    outpath = outpath + str(nb_layers) + "L/"

os.makedirs(outpath, exist_ok=True)

# MNIST/fashion - Loading
x_train, y_train, x_test, y_test = None, None, None, None
if dataset == 'MNIST':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
if dataset == 'FMNIST':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
if dataset == 'KMNIST':
    try:
        x_train = np.load('./kmnist-train-imgs.npz')['arr_0']
    except FileNotFoundError:
        print("ERROR: KMNIST files not found. You must download the kmnist dataset to this folder in the npz format.")
        exit()
    x_test = np.load('./kmnist-test-imgs.npz')['arr_0']
    y_train = np.load('./kmnist-train-labels.npz')['arr_0']
    y_test = np.load('./kmnist-test-labels.npz')['arr_0']

# Reshaping to flat digits
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# Normalizing to 0->1
maxval = np.max(np.abs(x_train))
x_train = x_train / maxval
x_test = x_test / maxval

# Taking 10,000 of training set as a validation set
# np.random.seed(42)
# valid_choice = np.random.choice(len(y_train), size=10000, replace=False)
# mask = np.zeros(len(y_train))
# mask[valid_choice] = 1
# x_valid, y_valid = x_train[mask == 1], y_train[mask == 1]
# x_train, y_train = x_train[mask != 1], y_train[mask != 1]

# Moving my datasets to the device
if is_cupy:
    onehot_y_train = indices_to_onehot(xp.asnumpy(y_train))
    onehot_y_test = indices_to_onehot(xp.asnumpy(y_test))
else:
    onehot_y_train = indices_to_onehot(y_train)
    onehot_y_test = indices_to_onehot(y_test)

device_x_train, device_y_train = xp.asarray(x_train), xp.asarray(onehot_y_train)
device_x_test, device_y_test = xp.asarray(x_test), xp.asarray(onehot_y_test)

net = None
# Constructing my network
if tower:
    net = TowerInvNet([x_train.shape[1], x_train.shape[1], 500, 200, 10], start_seed=seed, transfer_func=leaky_relu,
                      transfer_derivative_func=leaky_relu_derivative, transfer_inverse_func=leaky_relu_inverse,
                      orthogonal_init=ortho_init, adaptive=True, linear=linear)
else:
    net = SquareInvNet(x_train.shape[1], nb_layers, start_seed=seed, transfer_func=leaky_relu,
                       transfer_derivative_func=leaky_relu_derivative, transfer_inverse_func=leaky_relu_inverse,
                       orthogonal_init=ortho_init, adaptive=True, linear=linear)

if load:
    net.load_params(outpath)
    net.create_exact_inverse_model()
    t_accuracies = np.loadtxt(outpath + "training_acc.txt").tolist()
    t_losses = np.loadtxt(outpath + "training_loss.txt").tolist()
    test_accuracies = np.loadtxt(outpath + "test_acc.txt").tolist()
    test_losses = np.loadtxt(outpath + "test_loss.txt").tolist()
    t_accuracy, t_loss = t_accuracies[-1], t_losses[-1]
    test_accuracy, test_loss = test_accuracies[-1], test_losses[-1]

else:
    # Variables for storing progress
    t_accuracies, test_accuracies = [], []
    t_losses, test_losses = [], []
    # Initial accuracy and loss
    t_accuracy, t_loss = performance(net, device_x_train, device_y_train)
    test_accuracy, test_loss = performance(net, device_x_test, device_y_test)
    t_accuracies.append(t_accuracy), t_losses.append(t_loss)
    test_accuracies.append(test_accuracy), test_losses.append(test_loss)

# Training
batch_size = int(len(device_x_train) / 1000)
nb_batches_per_epoch = int(len(device_x_train) / batch_size)
print_interval = 10  # After how many batches should we print an update and collect stats

# Beginning training
print(outpath)
for e_index in range(nb_epochs):
    print("\nEpoch " + str(e_index))
    progress(0.0, t_accuracy)
    start_time = time.time()
    for b_index in range(nb_batches_per_epoch):
        # Run a forward and inverse pass
        forward = net.forward_pass(device_x_train[b_index * batch_size: (b_index + 1) * batch_size])
        targets = device_y_train[b_index * batch_size: (b_index + 1) * batch_size]
        targets = xp.hstack([targets, forward[-1][:, 10:]])

        # Computing the layer-wise errors and updating weights
        weight_updates, bias_updates = None, None
        if algorithm == 'BP':
            weight_updates, bias_updates = net.get_backprop_updates(forward, targets, ortho_weighting=ortho_reg)
        elif algorithm == 'GAIT':
            weight_updates, bias_updates = net.get_gait_updates(forward, targets, gamma=1e-3, ortho_weighting=ortho_reg)
        elif algorithm == 'TP':
            weight_updates, bias_updates = net.get_tp_updates(forward, targets, ortho_weighting=ortho_reg)

        net.update_forward_parameters(weight_updates, bias_updates, learning_rate)

        # Here we update the inverse model (using the cupy.inv function on every layer)
        if algorithm == 'GAIT' or algorithm == 'TP':
            if learn_inv:
                weight_updates, bias_updates = net.get_backward_updates(forward)
                net.update_backward_parameters(weight_updates, bias_updates, learning_rate)
            else:
                net.create_exact_inverse_model()

        # Upon some regular interval, check and print stats
        if (b_index != 0) and ((b_index % print_interval) == 0):
            t_accuracy, t_loss = performance(net, device_x_train, device_y_train)
            test_accuracy, test_loss = performance(net, device_x_test, device_y_test)
            t_accuracies.append(t_accuracy), t_losses.append(t_loss)
            test_accuracies.append(test_accuracy), test_losses.append(test_loss)
            progress(b_index / nb_batches_per_epoch, t_accuracy)
    print("\nTime: :" + str(time.time() - start_time))

print("\nTraining Complete!")

# Dumping Network State and Stats

# Dump training and validation progression
np.savetxt(outpath + "training_acc.txt", t_accuracies)
np.savetxt(outpath + "training_loss.txt", t_losses)
np.savetxt(outpath + "test_acc.txt", test_accuracies)
np.savetxt(outpath + "test_loss.txt", test_losses)

# Dump final synaptic weight matrices and biases for whole network
net.save_params(outpath)
