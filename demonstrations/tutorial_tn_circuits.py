r"""
.. _tn_circuits:

Tensor-Network Quantum Circuits
==============================

.. meta::
    :property="og:description": This demonstration explins how to simulate tensor-network quantum circuits.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tn_circuits.png

.. related::

   tutorial_plugins_hybrid Plugins and Hybrid computation
   tutorial_gaussian_transformation Gaussian transformation
   tutorial_state_preparation Training a quantum circuit with PyTorch

*Authors: Diego Guala, Esther Cruz-Rico, Shaoming Zhang, Juan Miguel Arrazola Last updated: 4 March 2022.*

This demonstration explains how to use PennyLane templates to design and implement tensor-network quantum circuits
as in `Huggins et. al. (2018) <https://arxiv.org/abs/1803.11537>`__. Tensor-network quantum circuits emulate the shape and connectivity of tensor networks such as matrix product states 
(MPS) and tree tensor networks (TTN).

We begin with a short introduction to tensor networks and explain their relationship to quantum circuits. Next, we 
demonstrate how PennyLane's templates make it easy to design, customize, and simulate these circuits. Finally, we
show how to use the templates to learn to classify the bars and stripes data set. This is a toy problem where the template
learns to recognize whether an image exhibits horizontal stripes or vertical bars.

Tensors and Tensor Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tensors are multi-dimensional arrays of numbers. 
Intuitively, they can be interpreted as a
generalization of scalars, vectors, and matrices. 
Tensors can be described by their rank, indices, and the dimension of the indices.
The rank is the number of indices in a tensor --- a scalar has 
rank zero, a vector has rank one, and a matrix has rank two.
The dimension of an index is how many values that index can take.
For example, a vector with three elements has one index that can take three
values. This is vector is therefore a rank one tensor and its index has dimension
three.

To define tensor networks, it is important to first understand tensor contration.
Two or more tensors can be contracted by summing over repeated indices.
In diagrammatic notation, the repeated indices appear as lines connecting tensors as in the figure below. 
We see two tensors of rank two, connected by one repeated index, :math:`k`. The dimension of the
repeated index is called the bond dimension.

.. image:: ../demonstrations/tn_circuits/Simple_TN_Color.PNG
    :align: center
    :width: 50 %

The contraction of the tensors above is equivalent to the standard 
matrix multiplication formula and can be expressed as

.. math::
    C_{ij} = \sum_{k}A_{ik}B_{kj},

where :math:`C_{ij}` denotes the entry for the :math:`i`-th row and :math:`j`-th column of the product :math:`C=AB`. 
Here, the number of terms in the summation is equal to the bond dimension of the index :math:`k`.

A tensor network is a collection of tensors where a subset of 
all indices are contracted. As mentioned above, we use diagrammatic notation
to specify which indices and tensors will be contracted together by connecting
individual tensors with lines.
Tensor networks can therefore represent complicated operations involving
several tensors with many indices contracted in sophisticated patterns.

Two well-known tensor network architectures are the MPS and TTN. These follow
specific patterns of connections between tensors and can be extended to have
many or few indices. Examples of these architectures with only a few tensors 
can be seen in the figure below.

.. image:: ../demonstrations/tn_circuits/MPS_TTN.PNG
    :align: center
    :width: 50 %

These tensor networks are commonly used to efficiently represent many-body quantum
systems in classical simulations (`Orus, 2014 <https://arxiv.org/abs/1306.2164>`)__.
We can reverse this, instead designing quantum systems, e.g., quantum circuits, that follow the structure and connectivity of tensor networks.
We call these circuits *tensor-network quantum circuits*.

In tensor-network quantum circuits, the tensor network architecture acts as a
guideline for the shape of the quantum circuit.
More specifically, the tensors in the tensor networks above are replaced with
unitary operations to obtain quantum circuits of the form:

.. image:: ../demonstrations/tn_circuits/MPS_TTN_Circuit_Color.PNG
    :align: center
    :width: 50 %

Since the unitary operations :math:`U_1` to :math:`U_3` are in principle completely general,
it is not always clear how to implement them with the gate sets available in quantum hardware. 
Instead, we can replace the unitary operations with variational quantum circuits,
determined by a specific template of choice.
The PennyLane tensor network templates allow us to do precisely this: implement tensor-network quantum
circuits with user-defined circuit ansatze as the unitary operations. In this sense, just as a template
is a strategy for arranging parametrized gates, tensor-network quantum circuits
are strategies for structuring circuit templates. They can therefore be interpreted as templates of templates,
i.e., as meta-templates. 



PennyLane Design
----------------
In this section, we demonstrate how to use PennyLane to build and simulate tensor-network quantum circuits.

Import PennyLane and Numpy
"""

import pennylane as qml
from pennylane import numpy as np

##############################################################################
# Build the circuit with PennyLane
#
# The following cell builds a quantum circuit in the shape of a matrix product state tensor network, and prints the result.
#
# Block defines a variational quantum circuit that takes the position of tensors in the network.


def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)


dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, expansion_strategy="device")
def circuit(template_weights):
    qml.MPS(
        wires=range(4),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3, 2])
print(f"The circuit result is {circuit(weights):.02f}")
qml.drawer.use_style("black_white")
qml.draw_mpl(circuit)(weights)

##############################################################################
# Easy circuit editing
#
#
# Using the :class:`~pennylane.MPS` template we can easily change the block type, depth, and size. For example:
# A deeper block

shape = qml.StronglyEntanglingLayers.shape(n_layers=2,n_wires =2)

def deep_block(weights, wires):
    qml.StronglyEntanglingLayers(weights,wires)


dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, expansion_strategy="device")
def circuit(template_weights):
    qml.MPS(
        wires=range(4),
        n_block_wires=2,
        block=deep_block,
        n_params_block=3,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=3))


weights = [np.random.random(size=shape)]*3
print(f"The circuit result is {circuit(weights):.02f}")
qml.draw_mpl(circuit)(weights)

##############################################################################
# A wider block


shapes = qml.SimplifiedTwoDesign.shape(n_layers=1, n_wires=4)
weights = [np.random.random(size=shape) for shape in shapes]

def large_block(weights, wires):
    qml.SimplifiedTwoDesign(initial_layer_weights=weights[0], weights=weights[1], wires=wires)


dev= qml.device('default.qubit',wires=8)

@qml.qnode(dev,expansion_strategy='device')
def circuit(template_weights):
    qml.MPS(wires=range(8),n_block_wires=4,block=large_block,n_params_block=2,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=7))

template_weights = [weights]*3
print(f'The circuit result is {circuit(template_weights):.02f}')
qml.draw_mpl(circuit)(template_weights)

##############################################################################
# A different tensor network
#
# We can also easily call a different tensor network architecture by using ``qml.TTN``:


def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RX(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)


dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, expansion_strategy="device")
def circuit(template_weights):
    qml.TTN(
        wires=range(4),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3, 2])
print(f"The circuit result is {circuit(weights):.02f}")
qml.draw_mpl(circuit)(weights)


##############################################################################
# Classifying the bars and stripes data set
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The bars and stripes data set is composed of binary black and white images of size :math:`n \times n` pixels,
# where either all pixels in any given column have the same color (bars) or all pixels in any given row have the same color (stripes),
# as shown in the image below.
#
# .. figure:: ../demonstrations/tn_circuits/BAS.png
#   :align: center
#   :height: 300
#
# For :math:`4\times 4` images, we can manually define this data set:

import matplotlib.pyplot as plt

BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
j = 1
for i in BAS:
    plt.subplot(1, 4, j)
    j += 1
    plt.imshow(np.reshape(i, [2, 2]), cmap="gray")

##############################################################################
# Then we define a tree tensor network quantum circuit that will learn to label these images


def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RX(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)


dev = qml.device("default.qubit", wires=4)
# this circuit gives a label, -1 to +1
@qml.qnode(dev, expansion_strategy="device")
def circuit(image, template_weights):
    qml.BasisStatePreparation(image, wires=range(4))
    qml.TTN(
        wires=range(4),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3, 2])
print(f"The circuit result is {circuit(BAS[0],weights):.02f}")
qml.draw_mpl(circuit)(BAS[0], weights)

##############################################################################
# We define a cost function and a gradient descent optimizer to train the circuit.

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
def costfunc(params):
    cost = 0
    for i in range(len(BAS)):
        if i<len(BAS)/2:
            cost+=circuit(BAS[i],params)
        else:
            cost-=circuit(BAS[i],params)
    return cost


##############################################################################
# We initialize the parameters and then train

np.random.seed(1)
params = np.random.random(size=[3, 2], requires_grad=True)

for k in range(100):
    params = optimizer.step(costfunc, params)

##############################################################################
# We print the results of the evaluation

for image in BAS:
    print(f"result is {circuit(image,params):.02f}")
    qml.draw_mpl(circuit)(image, params)
