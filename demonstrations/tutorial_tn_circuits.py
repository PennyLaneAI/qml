r"""
.. _tn_circuits:

Tensor Network Quantum Circuits
==============================

.. meta::
    :property="og:description": This demonstration explins how to simulate tensor network quantum circuits.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tn_circuits.png

.. related::

   tutorial_plugins_hybrid Plugins and Hybrid computation
   tutorial_gaussian_transformation Gaussian transformation
   tutorial_state_preparation Training a quantum circuit with PyTorch

*Author: PennyLane dev team. Last updated: 2 March 2022.*

This demonstration explains how to use PennyLane templates to design and implement tensor network quantum circuits
as in `Huggins et. al. (2018) <https://arxiv.org/abs/1803.11537>`__.

Background
------------

Tensor network quantum circuits emulate the shape and connectivity of tensor networks such as matrix product states 
(MPS) and tree tensor networks (TTN). By following tensor network architectures, quantum circuits can become easier
to simulate classically and provide a gradual transition from circuits that are classically simulable to circuits 
that require a quantum computer to evaluate (Huggins et. al., 2018)

We begin with a short introduction to tensor networks and explain their relationship to quantum circuits. Next, we 
demonstrate how PennyLane's templates make it easy to design, customize, and simulate these circuits. Finally, we
show how to use the templates to solve a toy machine learning problem.

Tensors and Tensor Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tensors are multi-dimensional arrays of numbers. Intuitively, they can be interpreted as a generalization of scalars, vectors, and matrices. 
Tensor can be described by their rank and bond dimension.
The rank is the number of indices in a tensor -- a scalar has rank zero, a vector has rank one, and a matrix has rank two.
Each index can take a certain number of values, we call this the bond dimension. A vector with three elements has a single index which can take three values.
This vector can be considered a tensor of rank one and bond dimension three.

To define tensor networks, it is important to first understand tensor contration.
Two or more tensors can be contracted by summing over repeated indices.
In diagrammatic notation, the repeated indices appear as lines connecting tensors as in the figure below. 
In this figure, we see two tensors of rank two, connected by one repeated index, :math:`k`.

.. image:: ../demonstrations/tn_circuits/Simple_TN.PNG
    :align: center
    :width: 80 %

The contraction of the tensors above is equivalent to the standard matrix multiplication formula and can be expressed as

.. math::
    C_{ij} = \sum_{k}A_{ik}B_{kj},

where :math:`C_{ij}` denotes the entry for the :math:`i`-th row and :math:`j`-th column of the product :math:`C=AB`. 
Here, the number of terms in the summation is equal to the bond dimension of the index :math:`k`.

A tensor network, then, is a collection of tensors where a subset of all indices are contracted based on the connections between tensors.
Tensor networks can therefore represent complicated operations involving several tensors with many indices contracted in sophisticated patterns.

Two well-known tensor network architectures are the MPS and TTN shown below.

.. image:: ../demonstrations/tn_circuits/MPS_TTN.PNG
    :align: center
    :width: 80 %

These tensor networks are commonly used to efficiently represent many-body quantum systems in classical simulations.
We can reverse this, instead designing quantum systems, i.e. quantum circuits, that follow the structure and connectivity of tensor networks.
We call these circuits *tensor network quantum circuits*.

In tensor network quantum circuits, the tensor network architecture acts a a meta-template for the quantum circuit.
In other words, the tensors in the tensor networks above are replaced with unitary operations to obtain quantum circuits of the form:

.. image:: ../demonstrations/tn_circuits/MPS_TTN_Circuit.PNG
    :align: center
    :width: 99 %

Since the unitary operations :math:`U_1` to :math:`U_3` are very general mathematical entities, it is not always clear how to implement them with
the gate sets available in quantum hardware. Instead, we can replace the unitary operations with circuit ansatze that perform a desired operation.
The PennyLane tensor network templates allow us to do precisely this, implement tensor network quantum circuits with user-defined circuit ansatze
as the unitary operations.

not sastisfied with the following paragraph:
In the following section, we demonstrate how to use PennyLane to build and simulate tensor network quantum circuits.
We will refer to the circuit ansatze that replace the unitary operations as *blocks*  and note that the bond dimension
of the tensor network, :math:`d`, is related to the number of wires per block, :math:`n`, as

.. math::
    d=2^{\frac{n}{2}}


PennyLane Design
----------------


Import PennyLane and Numpy
^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import pennylane as qml
from pennylane import numpy as np

##############################################################################
# Build the circuit with PennyLane
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The following cell builds a quantum circuit in the shape of a matrix product state tensor network and prints the result.
# 
# Block defines a variational quantum circuit that takes the position of tensors in the network.


def block(weights,wires):
    qml.CNOT(wires=wires)
    qml.RX(weights[0],wires=wires[0])
    qml.RY(weights[1],wires=wires[1])


dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev, expansion_strategy='device')
def circuit(template_weights):
    qml.MPS(wires=range(4),n_block_wires=2,block=block,n_params_block=2,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,2])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
qml.drawer.use_style('black_white')
qml.draw_mpl(circuit)(weights)

##############################################################################
# Easy circuit editing
# ^^^^^^^^^^^^^^^^^^^^ 
#
# Using the `qml.MPS` template we can easily change the block type, depth, and size. For example:
#
# A different block:
# ^^^^^^^^^^^^^^^^^^


def different_block(weights, wires):
    qml.CNOT(wires=wires)
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.Rot(weights[0],weights[1],weights[2],wires=wires[0])

dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev, expansion_strategy='device')
def circuit(template_weights):
    qml.MPS(wires=range(4),n_block_wires=2,block=different_block,n_params_block=3,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))

weights = np.random.random(size=[3,3])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
qml.draw_mpl(circuit)(weights)

##############################################################################
# A deeper block
# ^^^^^^^^^^^^^^


def deep_block(weights, wires):
    qml.SWAP(wires=wires)
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.Rot(weights[0],weights[1],weights[2],wires=wires[0])
    qml.Rot(weights[1],weights[2],weights[0],wires=wires[1])
    qml.SX(wires=wires[0])
    qml.RX(weights[3],wires=wires[0])
    qml.RY(weights[4],wires=wires[1])

dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev, expansion_strategy='device')
def circuit(template_weights):
    qml.MPS(wires=range(4),n_block_wires=2,block=deep_block,n_params_block=5,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,5])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
qml.draw_mpl(circuit)(weights)

##############################################################################
# A larger block
# ^^^^^^^^^^^^^^


def large_block(weights, wires):
    qml.MultiControlledX(control_wires=wires[0:3],wires=wires[3])
    qml.RX(weights[0],wires=wires[0])
    qml.RX(weights[1],wires=wires[1])
    qml.RX(weights[2],wires=wires[2])
    qml.RX(weights[3],wires=wires[3])

dev= qml.device('default.qubit',wires=8)

@qml.qnode(dev, expansion_strategy='device')
def circuit(template_weights):
    qml.MPS(wires=range(8),n_block_wires=4,block=large_block,n_params_block=5,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=7))


weights = np.random.random(size=[3,5])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
qml.draw_mpl(circuit)(weights)

##############################################################################
# A different tensor network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can also easily call a different tensor network architecture by using ``qml.TTN``:


def block(weights, wires):
    qml.CNOT(wires=wires)
    qml.RX(weights[0],wires=wires[0])
    qml.RX(weights[1],wires=wires[1])

dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev, expansion_strategy='device')
def circuit(template_weights):
    qml.TTN(wires=range(4),n_block_wires=2,block=block,n_params_block=2,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,2])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
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

BAS =[[1,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1]]
j=1
for i in BAS:
    plt.subplot(1,4,j)
    j+=1
    plt.imshow(np.reshape(i,[2,2]))

##############################################################################
# Then we define a tree tensor quantum circuit that will learn to label these images

def block(weights, wires):
    qml.CNOT(wires=wires)
    qml.RX(weights[0],wires=wires[0])
    qml.RX(weights[1],wires=wires[1])

dev= qml.device('default.qubit',wires=4)
#this circuit gives a label, -1 to +1
@qml.qnode(dev, expansion_strategy='device')
def circuit(image,template_weights):
    qml.BasisStatePreparation(image,wires=range(4))
    qml.TTN(wires=range(4),n_block_wires=2,block=block,n_params_block=2,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,2])
print(f'The circuit result is {circuit(BAS[0],weights):.02f}')
print('The circuit looks like:')
qml.draw_mpl(circuit)(BAS[0],weights)

##############################################################################
# We define a cost function and a gradient descent optimizer to train the circuit.

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
def costfunc(params):
    cost = 0
    cost += circuit(BAS[0],params)
    cost += circuit(BAS[1],params)
    cost -= circuit(BAS[2],params)
    cost -= circuit(BAS[3],params)
    return cost

##############################################################################
# We initialize the parameters and then train

np.random.seed(1)
params=np.random.random(size=[3,2], requires_grad=True)

for k in range(100):
    params = optimizer.step(costfunc, params)

##############################################################################
# We print the results of the evaluation

for image in BAS:
    print(f'result is {circuit(image,params):.02f}')
    print(qml.draw(circuit,expansion_strategy='device')(image,params))


