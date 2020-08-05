"""
.. _quantum_optical_neural_network:

Optimizing a quantum optical neural network
===========================================

.. meta::
    :property="og:description": Optimizing a quantum optical neural network using PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/qonn_thumbnail.png

This tutorial is based on a paper from `Steinbrecher et
al. (2019) <https://www.nature.com/articles/s41534-019-0174-7>`__ which
explores a Quantum Optical Neural Network (QONN) based on
Fock states.
Similar to the continuous-variable :doc:`quantum neural network </demos/quantum_neural_net>`
(CV QNN) model described by 
`Killoran et al. (2018) <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033063>`__, 
the QONN attempts to apply neural networks and deep learning theory to the quantum case,
using quantum data as well as a quantum hardware-based architecture.

We will focus on constructing a QONN as described in Steinbrecher et
al. and training it to work as a basic CNOT gate using a "dual-rail" state
encoding. This tutorial also provides a working example of how to use
third-party optimization libraries with PennyLane; in this case, 
`NLopt <https://nlopt.readthedocs.io/en/latest/>`__
will be used.

"""

######################################################################
# .. figure:: ../demonstrations/qonn/qonn_thumbnail.png
#     :width: 100%
#     :align: center
#
#     A quantum optical neural network using the Reck encoding (green)
#     with a Kerr non-linear layer (red)
#

######################################################################
# Background
# ----------
#
# The QONN is an optical architecture consisting of layers of linear
# unitaries, using the encoding described in `Reck et
# al. (1994) <https://dx.doi.org/10.1103/PhysRevLett.73.58>`__, and Kerr
# non-linearities applied on all involved optical modes. This setup can be
# constructed using arrays of beamsplitters and programmable phase shifts
# along with some form of Kerr non-linear material.
#
# By constructing a cost function based on the input-output relationship
# of the QONN, using the programmable phase-shift variables as
# optimization parameters, it can be trained to both act as an arbitrary
# quantum gate or to be able to generalize on previously unseen data. This
# is very similar to classical neural networks, and many classical machine
# learning task can in fact also be solved by these types of quantum deep
# neural networks.
#


######################################################################
# Code and simulations
# --------------------
#
# The first thing we need to do is to import PennyLane, NumPy, as well as an
# optimizer. Here we use a wrapped version of NumPy supplied by PennyLane
# which uses Autograd to wrap essential functions to support automatic
# differentiation.
#
# There are many optimizers to choose from. We could either use an
# optimizer from the ``pennylane.optimize`` module or we could use a
# third-party optimizer. In this case we will use the Nlopt library
# which has several fast implementations of both gradient-free and
# gradient-based optimizers.
#

import pennylane as qml
from pennylane import numpy as np

import nlopt


######################################################################
# We create a Strawberry Fields simulator device with as many quantum modes
# (or wires) as we want our quantum-optical neural network to have. Four
# modes are used for this demonstration, due to the use of a dual-rail encoding. The
# cutoff dimension is set to the same value as the number of wires (a
# lower cutoff value will cause loss of information, while a higher value
# might use unnecessary resources without any improvement).

dev = qml.device("strawberryfields.fock", wires=4, cutoff_dim=4)

######################################################################
#
# .. note::
#
#     You will need to have `S
#
# Create a layer function which def
trawberry Fields <https://strawberryfields.ai/>`__ as well as
#     the `Strawberry Fields plugin <https://pennylane-sf.readthedocs.io/en/latest/>`__
#     for PennyLane installed for this tutorial to work.
#

######################################################################
# Creating the QONN
# ~~~~~~~~~~~~~~~~~
#
# Create a layer function which defines one layer of the QONN, consisting
# of a linear
# `interferometer <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.subroutines.Interferometer.html>`__
# (i.e., an array of beamsplitters and phase shifts) and a non-linear
# Kerr interaction layer. Both the interferometer and the non-linear layer
# are applied to all modes. The triangular mesh scheme, described in `Reck
# et al. (1994) <https://dx.doi.org/10.1103/PhysRevLett.73.58>`__ is
# chosen here due to its use in the paper from Steinbrecher et al.,
# although any other interferometer scheme should work equally well.
# Some might even be slightly faster than the one we use here.
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
# measuring the mean photon number of each mode. The parameters to be
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
# Defining the cost function
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
# collects the outputs from the QONN (``predictions``) for each input
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
# .. figure:: ../demonstrations/qonn/cnot.png
#     :width: 30%
#     :align: center
# |
#
# We need to choose the inputs ``X`` and the corresponding labels ``Y``. They are
# defined using the dual-rail encoding, meaning that :math:`|0\rangle = [1, 0]` 
# (as a vector in the Fock basis of a single mode), and
# :math:`|1\rangle = [0, 1]`. So a CNOT transformation of :math:`|1\rangle|0\rangle = |10\rangle = [0, 1, 1, 0]`
# would give :math:`|11\rangle = [0, 1, 0, 1]`.
#
# Furthermore, we want to make sure that the gradient isn't calculated with regards
# to the inputs or the labels. We can do this by marking them with `requires_grad=False`.
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
# At this stage we could play around with other input-output
# combinations; just keep in mind that the input states should contain the
# same total number of photons as the ouput, since we want to use
# the dual-rail encoding. Also, since the QONN will
# act upon the states as a unitary operator, there must be a bijection
# between the inputs and the outputs, i.e., two different inputs must have
# two different outputs, and vice versa.
#


######################################################################
#
# .. note::
#     Other example gates we could use include the dual-rail encoded SWAP gate,
#
#     .. code:: python
#
#         X = np.array([[1, 0, 1, 0],
#                       [1, 0, 0, 1],
#                       [0, 1, 1, 0],
#                       [0, 1, 0, 1]])
#
#         Y = np.array([[1, 0, 1, 0],
#                       [0, 1, 1, 0],
#                       [0, 0, 0, 1],
#                       [0, 1, 0, 1]])
#
#     the single-rail encoded SWAP gate (remember to change the number of
#     modes to 2 in the device initialization above),
#
#     .. code:: python
#
#         X = np.array([[0, 1], [1, 0]])
#         Y = np.array([[1, 0], [0, 1]])
#
#     or the single 6-photon GHZ state (which needs 6 modes, and thus might be
#     very heavy on both memory and CPU):
#
#     .. code:: python
#
#         X = np.array([1, 0, 1, 0, 1, 0])
#         Y = (np.array([1, 0, 1, 0, 1, 0]) + np.array([1, 0, 1, 0, 1, 0])) / 2
#


######################################################################
# Now, we must set the number of layers to use and then calculate the
# corresponding number of initial parameter values, initializing them with
# a random value between :math:`-2\pi` and :math:`2\pi`. For the CNOT gate two layers is
# enough, although for more complex optimization tasks, many more layers
# might be needed. Generally, the more layers there are, the richer the
# representational capabilities of the neural network, and the better it
# will be at finding a good fit.
#
# The number of variables corresponds to the number of transmittivity
# angles :math:`\theta` and phase angles :math:`\phi`, while the Kerr
# non-linearity is set to a fixed strength.
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
#    [[ 3.09096645 -1.90233814 -6.04667805  5.9260723   0.6857962   4.22890046
#      -0.77881738 -1.9073714   0.99944676 -0.14188885 -6.04777972  3.1275478 ]
#     [ 2.20126372 -3.58195663 -3.57039035  5.38511235  4.52516263  3.34037724
#      -5.55181371 -3.63172916  0.87185867 -3.23167092  5.94563151  2.46618896]]

######################################################################
# The NLopt library is used for optimizing the QONN. For using
# gradient-based methods the cost function must be wrapped so that NLopt
# can access its gradients. This is done by calculating the gradient using
# autograd and then saving it in the ``grad[:]`` variable inside of the
# optimization function. The variables are flattened to conform to the
# requirements of both NLopt and the above-defined cost function.
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
#     Iter:    1    Cost: 4.4169e-01
#     Iter:    2    Cost: 3.0906e-01
#     Iter:    3    Cost: 2.6660e-01
#     Iter:    4    Cost: 1.8623e-01
#     Iter:    5    Cost: 1.3052e-01
#     Iter:    6    Cost: 6.4911e-02
#     Iter:    7    Cost: 2.7688e-02
#     Iter:    8    Cost: 1.3922e-02
#     Iter:    9    Cost: 4.7962e-03
#     Iter:   10    Cost: 2.8119e-03
#     Iter:   11    Cost: 8.2535e-04
#     Iter:   12    Cost: 2.9660e-04
#     Iter:   13    Cost: 9.5267e-05
#     Iter:   14    Cost: 3.1653e-05
#     Iter:   15    Cost: 1.0486e-05
#     Iter:   16    Cost: 3.5410e-06
#     Iter:   17    Cost: 1.2191e-06
#     Iter:   18    Cost: 4.3391e-07
#     Iter:   19    Cost: 1.6125e-07
#     Iter:   20    Cost: 6.2801e-08
#     Iter:   21    Cost: 2.5614e-08
#     Iter:   22    Cost: 1.0838e-08
#     Iter:   23    Cost: 4.5478e-09
#     Iter:   24    Cost: 1.8192e-09
#     Iter:   25    Cost: 7.1631e-10
#     Iter:   26    Cost: 2.9395e-10
#     Iter:   27    Cost: 1.3205e-10
#     Iter:   28    Cost: 6.2740e-11
#     Iter:   29    Cost: 2.7242e-11
#     Iter:   30    Cost: 1.0773e-11


######################################################################
# .. note::
#
#     It’s also possible to use any of PennyLane’s built-in 
#     gradient-based optimizers:
#
#     .. code:: python
#
#         from pennylane.optimize import AdamOptimizer
#
#         opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)
#
#         var = var_init
#         for it in range(200):
#             var = opt.step(lambda v: cost(v, X, Y), var)
#
#             if (it+1) % 20 == 0:
#                 print(f"Iter: {it+1:5d} | Cost: {cost(var, X, Y):0.7f} ")
#


######################################################################
# Finally, we print the results.
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
#     The optimized parameters (layers, parameters):
#     [[ 3.80632856 -1.97708172 -6.28318531  5.83698132  0.98132917  4.11752553
#       -0.45925719 -1.49556726  0.99504518 -0.01590272 -6.28318531  2.53147249]
#     [ 2.21899793 -3.12552777 -3.14114951  5.62958181  3.91498688  3.91493446
#       -5.97854181 -3.68357379  0.87626027 -2.85015063  6.28318531  2.66192377]]
#
#     [1 0 1 0] --> [1. 0. 1. 0.], should be [1 0 1 0]
#     [1 0 0 1] --> [1. 0. 0. 1.], should be [1 0 0 1]
#     [0 1 1 0] --> [0. 1. 0. 1.], should be [0 1 0 1]
#     [0 1 0 1] --> [0. 1. 1. 0.], should be [0 1 1 0]

##############################################################################
# We can also print the circuit to see how the final network looks.

quantum_neural_net(var_init, X[0])
print(quantum_neural_net.draw())

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     0: ──|1⟩───────────────────────────────────────────╭BS(-6.047, 0.999)───R(0.0)──────────────Kerr(1.571)──────────────────────────────────────────────────────────────────────────╭BS(-3.57, 0.872)────R(0.0)─────────────Kerr(1.571)────────────────────────────────┤ ⟨n⟩
#     1: ──|0⟩──────────────────────╭BS(-1.902, -1.907)──╰BS(-6.047, 0.999)──╭BS(0.686, -6.048)───R(0.0)────────────Kerr(1.571)───────────────────────────────────╭BS(-3.582, -3.632)──╰BS(-3.57, 0.872)───╭BS(4.525, 5.946)───R(0.0)───────────Kerr(1.571)───────────────┤ ⟨n⟩
#     2: ──|1⟩──╭BS(3.091, -0.779)──╰BS(-1.902, -1.907)──╭BS(5.926, -0.142)──╰BS(0.686, -6.048)──╭BS(4.229, 3.128)──R(0.0)───────Kerr(1.571)──╭BS(2.201, -5.552)──╰BS(-3.582, -3.632)──╭BS(5.385, -3.232)──╰BS(4.525, 5.946)──╭BS(3.34, 2.466)──R(0.0)───────Kerr(1.571)──┤ ⟨n⟩
#     3: ──|0⟩──╰BS(3.091, -0.779)───────────────────────╰BS(5.926, -0.142)──────────────────────╰BS(4.229, 3.128)──R(0.0)───────Kerr(1.571)──╰BS(2.201, -5.552)───────────────────────╰BS(5.385, -3.232)─────────────────────╰BS(3.34, 2.466)──R(0.0)───────Kerr(1.571)──┤ ⟨n⟩
#
