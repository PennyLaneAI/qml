"""
.. _quantum_optical_neural_network:

Quantum Optical Neural Network (QONN)
=====================================

"""


######################################################################
# Imports
# -------
#
# The first thing to do is to import PennyLane, a wrapped version of
# NumPy, and an optimizer.
#

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer


######################################################################
# Create a Strawberry Fields simulator device with as many quantum modes
# (or wires) that you wish your quantum optical neural network to have. 4
# modes are used for this demonstration. You will need to have the
# Strawberry Fields plugin for PennyLane installed.
#

dev = qml.device("strawberryfields.fock", wires=4, cutoff_dim=4)


######################################################################
# Quantum node
# ------------
#
# Create a layer function which defines one layer of the QONN, consisting
# of a linear interferometer (i.e. an array of beamsplitters and phase
# shifters) and a non-linear Kerr interaction layer. Both the
# interferometer and the non-linear layer are applied to all 4 modes.
#
# **Note**: while the interferometer needs to be applied on all modes at
# the same time, the non-linear Kerr layer need to be applied on each mode
# one-at-a-time.
#

def layer(theta, phi, wires):
    M = len(wires)
    phi_nonlinear = np.ones(M) * np.pi

    qml.templates.interferometer.Interferometer(
        theta, phi, np.ones(M),
        wires=wires,
        mesh='rectangular',
        beamsplitter='clements'
    )

    [qml.Kerr(phi_nonlinear[i], wires=i) for i in wires]


######################################################################
# Define the full QONN by building each layer one-by-one and then
# returning the mean photon number of each mode.
#

@qml.qnode(dev)
def quantum_neural_net(var, x=None):
    wires = list(range(len(x)))

    # Encode input x into a sequence of quantum fock states
    [qml.FockState(x[i], wires=i) for i in wires]

    # Not sure why this doesn't work.
    # qml.FockStateVector(x, wires=wires)

    # "layer" subcircuits
    for i, v in enumerate(var):
        layer(v[:len(v)//2], v[len(v)//2:], wires)

    return [qml.expval(qml.NumberOperator(w)) for w in wires]


######################################################################
# Objective
# ---------
#
# Define a helper function to calculate the normalized square loss of two
# vectors. The square loss function returns a value between 0 and 1, where
# 0 means that ``labels`` and ``predictions`` are equal and 1 means that
# the vectors are fully orthogonal.
#

def square_loss(labels, predictions):
    term = 0
    for l, p in zip(labels, predictions):
        lnorm = l / np.linalg.norm(l)
        pnorm = p / np.linalg.norm(p)

        term += np.abs(lnorm @ pnorm.T) ** 2

    return 1 - term / len(labels)


######################################################################
# Next, define the cost function to be used during the optimization. It
# gets the outputs of the neural (``predicitions``) based on a set of
# inputs (``data_inputs``) and then calculates the square loss between the
# predictions and the wanted outputs (``labels``).
#

def cost(var, data_input, labels):
    predictions = np.array([quantum_neural_net(var, x=x) for x in data_input])
    sl = square_loss(labels, predictions)

    return sl


######################################################################
# Optimization
# ------------
#
# For this optimization we will train the network to perform as a CNOT
# gate. That is, it should transform the input states in the following
# way:
#
# *insert-nice-image-of-cnot-gate-fock-state-transformations-here*
#

# Define the CNOT input-output states

#X = np.array([[1, 0, 1, 0],
#              [1, 0, 0, 1],
#              [0, 1, 1, 0],
#              [0, 1, 0, 1]])

#Y = np.array([[1, 0, 1, 0],
#              [1, 0, 0, 1],
#              [0, 1, 0, 1],
#              [0, 1, 1, 0]])


# The swap gate over two modes

X = np.array([[0, 1], [1, 0]])
Y = np.array([[1, 0], [0, 1]])



######################################################################
# Determining the number of layers to use and calculate the corresponding
# number of initial values to use.
#

num_layers = 2
M = len(X[0])
num_variables_per_layer = M * (M - 1)

var_init = (4 * np.random.rand(num_layers, num_variables_per_layer) - 2) * np.pi
print(var_init)


######################################################################
# Use the Adam optimizer to iterate through the optimization step-by-step,
# updating the variables each time.
#

import autograd
import nlopt

cost_grad = autograd.grad(cost)

print_every = 1

# Wrap the cost so that NLopt can use it for gradient-based optimizations
evals = 0
def cost_wrapper(var, grad=[]):
    global evals
    evals += 1

    if grad.size > 0:
        grad[:] = cost_grad(var.reshape((num_layers, num_variables_per_layer)), X, Y).flatten()
    cost_val = cost(var.reshape((num_layers, num_variables_per_layer)), X, Y)

    if evals % print_every == 0:
        print("Iter: {:4d}    Cost: {:.4e}".format(evals, cost_val))

    return cost_val


# Choose an algorithm
opt_algorithm = nlopt.LD_LBFGS  # Gradient-based
# opt_algorithm = nlopt.LN_BOBYQA  # Gradient-free

opt = nlopt.opt(opt_algorithm, num_layers * num_variables_per_layer)

opt.set_min_objective(cost_wrapper)

opt.set_lower_bounds(-2 * np.pi * np.ones(num_layers * num_variables_per_layer))
opt.set_upper_bounds(2 * np.pi * np.ones(num_layers * num_variables_per_layer))

var = opt.optimize(var_init.flatten())
var = var.reshape(var_init.shape)


######################################################################
# It's also possible to use any of PennyLane's built-in optimizers.
#

# opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

# var = var_init
# for it in range(200):
#     var = opt.step(lambda v: cost(v, X, Y), var)

#     # Nice to print cost here, although the expensive `cost` function needs to be called again.
#     if (it + 1) % 20 == 0:
#         print("Iter: {:5d} | Cost: {:0.7f} ".format(it + 1, cost(var, X, Y)))


######################################################################
# Finally, print the results.
#

print("The optimized parameters:\n {}".format(var))

Y_pred = np.array([quantum_neural_net(var, x=x) for x in X])
for i, x in enumerate(X):
    print("{} --> {}, {}".format(x, Y_pred[i], Y[i]))
