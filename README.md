# GAIT Propagation Code Instructions
This code accompanies the arXiv pre-print titled ["**GAIT-prop: A biologically plausible learning rule derived from backpropagation of error**"](https://arxiv.org/abs/2006.06438).

Enclosed ar a few files, a brief description of each:
- *run.py*: A python script designed to instantiate, train, and dump invertible neural network models. This is the main python file for model reproduction.
- *inv_layer.py*: The class definition for an invertible (non-)linear layer of neurons including a forward and inverse pass definition
- *inv_net.py*: The class definitions of a `SquareInvNet` class which produces networks of fixed layer-width and `TowerInvNet` providing equivalent functionality for networks with variable (strictly decreasing) hidden layer widths.
- *utils.py*: Miscellaneous python functions useful for the invertible network simulations
- *conda_requirements.py* -- A dump of the python environment used to produce the results in this paper. Note that some of these packages (such as cupy 7.4.0) were installed via pip. This file should allow reproduction of the python environment. 

**run.py** accepts a number of command-line arguments.
These are detailed below.
If cupy is not installed, run.py defaults to numpy.
Note however that this is a MUCH slower mode of operation.

Long Arguments:

|option|default|explanation|
|---|---|---|
|`--algorithm={BP,GAIT,TP}`|BP|The training algorithm used.|
|`--seed=`|1|The seed for the random generators which produce network weights|
|`--orthogona_reg=`|0.0|The strength of the orthogonal regularizer (lambda)|
|`--device=`|0|If using cupy, the GPU device ID for simulation|
|`--nb_epochs=`|100|The number of training epochs to run (after which accuracies and network parameters are saved)|
|`--nb_layers=`|1|The number of hidden layers to simulate a network with (only applicable to Square networks|
|`--learning_rate=`|0.0001|The learning rate to use for the simulation|
|`--dataset={MNIST,KMNIST,FMNIST}`|MNIST|The dataset to use for training|


Short Arguments:


|option|explanation|
|---|---|
| `--linear` |Makes use of a linear transfer function instead of leaky-ReLu.|
| `--tower` |Instead of creating a Square (fixed-width) network, a network with different sized layers is used. The shape of the layers is fixed in run.py (L153)|


A few examples of command-line executions are provided
    
    # Full-Width Networks
    # Training a four hidden-layer network by BP/GAIT/TP (with parameters identified in the paper):
    ./run.py --algorithm=BP --learning_rate=0.0001 --nb_layers=4 --nb_epochs=100
    ./run.py --algorithm=GAIT --learning_rate=0.0001 --orthogonal_reg=0.1 --nb_layers=4 --nb_epochs=100
    ./run.py --algorithm=TP --learning_rate=0.00001 --orthogonal_reg=1000.0 --nb_layers=4 --nb_epochs=100
    
    # Training a variable width network by GAIT-prop:
    ./run.py --algorithm=GAIT --learning_rate=0.0001 --orthogonal_reg=0.1 --nb_layers=4 --nb_epochs=100 --tower
    
    # Training a linear TP network with four hidden layers
    ./run.py --algorithm=TP --linear --learning_rate=0.00001 --orthogonal_reg=1000.0 --nb_layers=4 --nb_epochs=100


