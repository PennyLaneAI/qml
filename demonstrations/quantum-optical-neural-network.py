"""
.. _quantum_optical_neural_network:

Optimizing Quantum Optical Neural Networks (QONN) using NLopt
=============================================================

This tutorial is based on the paper from `Steinbrecher et
al. (2019) <https://www.nature.com/articles/s41534-019-0174-7>`__ which
explores a fock-state based Quantum Optical Neural Network (QONN).
Similar to the continuous-variable quantum neural network (CV QNN) model
described by `Killoran et
al. (2018) <https://arxiv.org/abs/1806.06871>`__, the QONN attempts to
apply neural networks and deep learning theory to the quantum case;
using both quantum data as well as quantum hardware based simulation
methods.

We will focus on constructing a QONN as described in Steinbrecher et
al. and training it to work as a basic CNOT gate using dual-rail state
encodings. Since these simulations are fairly heavy, the third-party
optimization library `NLopt <https://nlopt.readthedocs.io/en/latest/>`__
will be used.

"""


######################################################################
# Background
# ----------
#
# The QONN is an optical architecture consisting of layers of linear
# unitaries, using the encoding described in `Reck et
# al. (1994) <https://dx.doi.org/10.1103/PhysRevLett.73.58>`__, and Kerr
# non-linearities applied on all involved modes. This setup can be
# constructed using arrays of beamsplitters and programmable phaseshifts
# along with some form of Kerr non-linear material.
#
# By constructing a cost function based on the input-output relationship
# of the QONN, using the programmable phaseshifts variables as
# optimization parameters, it can be trained to both act as an artbitrary
# quantum gate or to be able to generalize on previously unseen data. This
# is very similar to classical neural networks, and many classical machine
# learning task can in fact also be solved on these type of quantum deep
# neural networks.
#


######################################################################
# Code and simulations
# --------------------
#
# The first thing we need to do is to import PennyLane, NumPy and an
# optimizer. Here we use a wrapped version of NumPy supplied by PennyLane
# which uses Autograd to wrap essential functions to support automatic
# differentiation.
#
# There are many optimizers to choose from. We could either use an
# optimizer from the ``pennylane.optimize`` module or we could use a
# third-party optimizer. In this case we will mainly use the Nlopt library
# which has several fast implementations of both gradient-free and
# gradient-based optimizers.
#

import pennylane as qml
from pennylane import numpy as np

import nlopt


######################################################################
# Create a Strawberry Fields simulator device with as many quantum modes
# (or wires) that you wish your quantum optical neural network to have. 4
# modes are used for this demonstration due to the dual-rail encoding. The
# cuttof dimension should be set to the same as the number of wires (a
# lower cutoff value will cause loss of information, while a higher value
# will only use unnecessary resources without any improvement).
#
# .. note::
#
#     You will need to have `Strawberry Fields <https://strawberryfields.ai/>`__ as well as
#     the `Strawberry Fields plugin <https://pennylane-sf.readthedocs.io/en/latest/>`__
#     for PennyLane installed for this tutorial to work.
#

dev = qml.device("strawberryfields.fock", wires=4, cutoff_dim=4)


######################################################################
# Creating the QONN
# ~~~~~~~~~~~~~~~~~
#
# Create a layer function which defines one layer of the QONN, consisting
# of a linear
# `interferometer <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.subroutines.Interferometer.html>`__
# (i.e. an array of beamsplitters and phase shifters) and a non-linear
# Kerr interaction layer. Both the interferometer and the non-linear layer
# are applied to all modes. The triangular mesh scheme, described in `Reck
# et al. (1994) <https://dx.doi.org/10.1103/PhysRevLett.73.58>`__ is
# chosen here due to its use in the paper from Steinbrecher et al.,
# although any other interferometer scheme should work equally well, and
# some might even be slightly faster.
#
# **Note**: while the interferometer must be applied on all modes at the
# same time, the non-linear Kerr layer need to be applied on each mode
# one-at-a-time.
#

def layer(theta, phi, wires):
    M = len(wires)
    phi_nonlinear = np.pi / 2

    qml.templates.Interferometer(
        theta, phi, np.zeros(M), wires=wires, mesh="triangular",
    )

    for i in wires:
        qml.Kerr(phi_nonlinear, wires=i)


######################################################################
# Next, we define the full QONN by building each layer one-by-one and then
# returning the mean photon number of each mode. The parameters to be
# optimized are all contained in ``var``, where each element in ``var`` is
# a list of parameters ``theta`` and ``phi`` for a specific layer.
#

@qml.qnode(dev)
def quantum_neural_net(var, x):
    wires = list(range(len(x)))

    # Encode input x into a sequence of quantum fock states
    for i in wires:
        qml.FockState(x[i], wires=i)

    # "layer" subcircuits
    for i, v in enumerate(var):
        layer(v[: len(v) // 2], v[len(v) // 2 :], wires)

    return [qml.expval(qml.NumberOperator(w)) for w in wires]


######################################################################
# Defining the cost-function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A helper function is needed to calculate the normalized square loss of
# two vectors. The square loss function returns a value between 0 and 1,
# where 0 means that ``labels`` and ``predictions`` are equal and 1 means
# that the vectors are fully orthogonal.
#

def square_loss(labels, predictions):
    term = 0
    for l, p in zip(labels, predictions):
        lnorm = l / np.linalg.norm(l)
        pnorm = p / np.linalg.norm(p)

        term = term + np.abs(np.dot(lnorm, pnorm.T)) ** 2

    return 1 - term / len(labels)


######################################################################
# Finally, we define the cost function to be used during optimization. It
# collects the outputs from the QONN (``predictions``) for each input in
# (``data_inputs``) and then calculates the square loss between the
# predictions and the true outputs (``labels``).
#

def cost(var, data_input, labels):
    predictions = np.array([quantum_neural_net(var, x) for x in data_input])
    sl = square_loss(labels, predictions)

    return sl


######################################################################
# Optimizing for the CNOT gate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For this tutorial we will train the network to function as a CNOT gate.
# That is, it should transform the input states in the following way:
#
# *insert-nice-image-of-cnot-gate-fock-state-transformations-here*
#
# We need to choose the inputs ``X`` and their labels ``Y``. They are
# defined using dual-rail encoding, meaning that :math:`|0\rangle = [1, 0]` and
# `|1\rangle = [0, 1]`, e.g. a CNOT transformation of `|10\rangle = [0, 1, 1, 0]`
# would be `|11\rangle = [0, 1, 0, 1]`.
#

# Define the CNOT input-output states (dual-rail encoding) and initialize
# them as non-differentiable.

X = np.array([[1, 0, 1, 0],
              [1, 0, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 0, 1]], requires_grad=False)

Y = np.array([[1, 0, 1, 0],
              [1, 0, 0, 1],
              [0, 1, 0, 1],
              [0, 1, 1, 0]], requires_grad=False)


######################################################################
# At this stage you could play around with other input-output
# combinations; just keep in mind that the input states should contain the
# same total number of photons as the ouput, since no photons will be
# created or lost (assuming loss-less circuits). Also, since the QONN will
# act upon the states as a unitary operator, there must be a bijection
# between the inputs and the outputs, i.e. two different inputs must have
# two different outputs, and vice versa.
#


######################################################################
# Examples include the dual-rail encoded SWAP gate:
#
# .. code:: python
#
#    X = np.array([[1, 0, 1, 0],
#                  [1, 0, 0, 1],
#                  [0, 1, 1, 0],
#                  [0, 1, 0, 1]])
#
#    Y = np.array([[1, 0, 1, 0],
#                  [0, 1, 1, 0],
#                  [0, 0, 0, 1],
#                  [0, 1, 0, 1]])
#
# the single-rail encoded SWAP gate (remember to change the number of
# modes to 2 in the device initialization above):
#
# .. code:: python
#
#    X = np.array([[0, 1], [1, 0]])
#    Y = np.array([[1, 0], [0, 1]])
#
# or the single 6-photon GHZ state (which needs 6 modes, and thus might be
# very heavy on both memory and CPU):
#
# .. code:: python
#
#    X = np.array([1, 0, 1, 0, 1, 0])
#    Y = (np.array([1, 0, 1, 0, 1, 0]) + np.array([1, 0, 1, 0, 1, 0])) / 2
#


######################################################################
# Now, we must set the number of layers to use and then calculate the
# corresponding number of initial parameter values, initializing them with
# a random value between :math:`-2\pi` and :math:`2\pi`. For the CNOT gate 2 layers is
# enough, although for more complex optimization tasks, many more layers
# might be needed. Generally, the more layers there are, the better the neural
# network will be at finding a solution, including having better generalization
# power for predicting on un-trained data.
#
# The number of variables corresponds to the number of transmittivity
# angles :math:`\theta` and the same number of phase angles :math:`\phi`, while the Kerr
# non-linearity is set to full strength.
#

num_layers = 2
M = len(X[0])
num_variables_per_layer = M * (M - 1)

var_init = (4 * np.random.rand(num_layers, num_variables_per_layer) - 2) * np.pi
print(var_init)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#    [[ 4.41249606 -0.57430797  6.09429979  2.42169997 -1.34364061 -3.40119641
#      -1.2612163   4.17884671 -5.75922559 -4.42392391  3.7409752   0.88462534]
#     [-0.47474198 -3.77282506  4.53681897 -3.61406824 -3.56062435 -2.5747048
#       1.11327766  1.76360541 -2.89212374 -6.02659853 -5.157448   -4.01601384]]


######################################################################
# The NLopt library is used for optimizing the QONN. For using
# gradient-based methods the cost function must be wrapped so that NLopt
# can access its gradients. This is done by calculating the gradient using
# autograd and then saving it in the ``grad[:]`` variable inside of the
# optimization function. The variables are flattened to conform to the
# requirements of both NLopt and the above defined cost function.
#

cost_grad = qml.grad(cost)

print_every = 1

# Wrap the cost so that NLopt can use it for gradient-based optimizations
evals = 0
def cost_wrapper(var, grad=[]):
    global evals
    evals += 1

    if grad.size > 0:
        # Get the gradient for `var` (idx 0) by first "unflattening" it
        var_grad = cost_grad(var.reshape((num_layers, num_variables_per_layer)), X, Y)[0]
        grad[:] = var_grad.flatten()
    cost_val = cost(var.reshape((num_layers, num_variables_per_layer)), X, Y)

    if evals % print_every == 0:
        print(f"Iter: {evals:4d}    Cost: {cost_val:.4e}")

    return float(cost_val)


# Choose an algorithm
opt_algorithm = nlopt.LD_LBFGS  # Gradient-based
# opt_algorithm = nlopt.LN_BOBYQA  # Gradient-free

opt = nlopt.opt(opt_algorithm, num_layers*num_variables_per_layer)

opt.set_min_objective(cost_wrapper)

opt.set_lower_bounds(-2*np.pi * np.ones(num_layers*num_variables_per_layer))
opt.set_upper_bounds(2*np.pi * np.ones(num_layers*num_variables_per_layer))

var = opt.optimize(var_init.flatten())
var = var.reshape(var_init.shape)


##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#    Iter:    1    Cost: 8.7560e-01
#    Iter:    2    Cost: 7.5588e-01
#    Iter:    3    Cost: 6.1698e-01
#    Iter:    4    Cost: 5.3337e-01
#    Iter:    5    Cost: 3.8656e-01
#    Iter:    6    Cost: 3.4314e-01
#    Iter:    7    Cost: 2.7050e-01
#    Iter:    8    Cost: 1.2037e-01
#    Iter:    9    Cost: 2.8259e-02
#    Iter:   10    Cost: 6.4414e-02
#    Iter:   11    Cost: 1.4495e-02
#    Iter:   12    Cost: 7.0265e-03
#    Iter:   13    Cost: 3.9245e-03
#    Iter:   14    Cost: 1.4525e-03
#    Iter:   15    Cost: 7.1931e-04
#    Iter:   16    Cost: 3.2824e-04
#    Iter:   17    Cost: 1.5034e-04
#    Iter:   18    Cost: 7.1917e-05
#    Iter:   19    Cost: 2.9580e-05
#    Iter:   20    Cost: 1.1786e-05
#    Iter:   21    Cost: 4.8228e-06
#    Iter:   22    Cost: 2.1685e-06
#    Iter:   23    Cost: 1.1014e-06
#    Iter:   24    Cost: 6.1727e-07
#    Iter:   25    Cost: 3.4732e-07
#    Iter:   26    Cost: 1.7053e-07
#    Iter:   27    Cost: 7.8882e-08
#    Iter:   28    Cost: 4.9757e-08
#    Iter:   29    Cost: 3.3791e-08
#    Iter:   30    Cost: 2.6556e-08
#    Iter:   31    Cost: 2.3685e-08
#    Iter:   32    Cost: 1.9864e-08
#    Iter:   33    Cost: 1.6450e-08
#    Iter:   34    Cost: 1.4705e-08
#    Iter:   35    Cost: 1.2818e-08
#    Iter:   36    Cost: 1.1974e-08
#    Iter:   37    Cost: 1.1177e-08
#    Iter:   38    Cost: 1.0169e-08
#    Iter:   39    Cost: 9.2866e-09
#    Iter:   40    Cost: 8.4649e-09
#    Iter:   41    Cost: 7.5698e-09
#    Iter:   42    Cost: 6.5851e-09
#    Iter:   43    Cost: 5.2888e-09
#    Iter:   44    Cost: 3.5282e-09
#    Iter:   45    Cost: 1.6706e-09
#    Iter:   46    Cost: 6.2609e-10
#    Iter:   47    Cost: 2.2746e-10
#    Iter:   48    Cost: 7.9969e-11
#    Iter:   49    Cost: 2.9002e-11
#    Iter:   50    Cost: 1.0558e-11


######################################################################
# It’s also possible to use any of PennyLane’s built-in optimizers,
# supporting both gradient-based and gradient-free optimization methods:
#
# .. code:: python
#
#    from pennylane.optimize import AdamOptimizer
#
#    opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)
#
#    var = var_init
#    for it in range(200):
#        var = opt.step(lambda v: cost(v, X, Y), var)
#
#        # Nice to print cost here, although the expensive `cost` function needs to be called again.
#        if (it+1) % 20 == 0:
#            print(f"Iter: {it+1:5d} | Cost: {cost(var, X, Y):0.7f} ")
#


######################################################################
# Finally, print the results.
#

print(f"The optimized parameters (layers, parameters):\n {var}\n")

Y_pred = np.array([quantum_neural_net(var, x) for x in X])
for i, x in enumerate(X):
    print(f"{x} --> {Y_pred[i].round(2)}, should be {Y[i]}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#    The optimized parameters (layers, parameters):
#    [[ 5.04634589 -0.93428756  6.28318531  2.1400553  -0.80027272 -2.85121686
#      -1.33078852  4.04343622 -5.94204607 -4.34585754  4.49141261  0.78578627]
#     [-1.56922751 -2.78848283  3.14133072 -3.14770991 -5.82818671 -3.92929427
#       2.01403228  1.34968129 -2.70930324 -6.28318531 -5.3585575  -3.8178278 ]]
#
#    [1 0 1 0] --> [1. 0. 1. 0.], should be [1 0 1 0]
#    [1 0 0 1] --> [1. 0. 0. 1.], should be [1 0 0 1]
#    [0 1 1 0] --> [0. 1. 0. 1.], should be [0 1 0 1]
#    [0 1 0 1] --> [0. 1. 1. 0.], should be [0 1 1 0]
