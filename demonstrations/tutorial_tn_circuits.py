r"""
.. _tn_circuits:

Tensor Network Quantum Circuits
==============================

.. meta::
    :property="og:description": This demonstration introduces how to use PennyLane to design and implement tensor network shaped quantum circuits.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tn_circuits.png

.. related::

   tutorial_plugins_hybrid Plugins and Hybrid computation
   tutorial_gaussian_transformation Gaussian transformation
   tutorial_state_preparation Training a quantum circuit with PyTorch

*Author: PennyLane dev team. Last updated: 16 Feb 2022.*

This demonstration introduces how to use PennyLane to design and implement tensor network shaped quantum circuits.

We begin with a short introduction to tensor networks, explain their relationship to quantum circuits, 
give some examples of how to implement tensor network quantum circuits in PennyLane, and then solve a toy problem.
Tensor network quantum circuits are inspired by `Huggins (2018) <https://arxiv.org/abs/1803.11537>`__.

Introduction
-------------------
We begin with a short introduction to tensor networks, explain their relationship to quantum circuits, 
give some examples of how to implement tensor network quantum circuits in PennyLane, and then solve a toy problem.

Tensor Networks
-------------------
Tensors are multi-dimensional arrays of numbers. Intuitively, they can be interpreted as a generalization of scalars, vectors, and matrices. The rank is the number of indices in a tensor -- a scalar has rank zero, a vector has rank one, and a matrix has rank two. A group of tensors can be contracted by summing over repeated indices. For example, the standard matrix multiplication formula can be expressed as a tensor contraction

.. math::
    C_{ij} = \sum_{k}A_{ik}B_{kj},

where $C_{ij}$ denotes the entry for the $i$-th row and $j$-th column of the product $C=AB$.

A tensor network is a collection of tensors where a subset of all indices are contracted according to some specified rule. Tensor networks can therefore represent complicated operations involving several tensors with many indices contracted in sophisticated patterns.

Two well-known tensor network architectures are matrix product states (MPS) and tree tensor networks (TTN) shown below.

.. figure:: ../demonstrations/tn_circuits/MPS.png
    :align: center
    :width: 90%

.. figure:: ../demonstrations/tn_circuits/TTN.png
    :align: center
    :width: 90%

It is possible to design quantum circuits that follow the structure and connectivity of these and other tensor networks. We call these quantum circuits <em>tensor network quantum circuits</em>.

In this case, the tensor network architecture acts a a meta-template for the quantum circuit. The tensors in the tensor networks above are replaced with unitary operations to obtain quantum circuits of the form:

.. figure:: ../demonstrations/tn_circuits/MPS_Circuit.png
    :align: center
    :width: 90%
    
.. figure:: ../demonstrations/tn_circuits/TTN_Circuit.png
    :align: center
    :width: 90%

The TTN image should have the initial empty wires erased. The initial bubbles v1-v4 could be color coded to match the circuit following


We can define some equivalencies between the tensor networks and the quantum circuits. For example, in a tensor network, the bond dimension 
is the number of values that a connection can take (not clear). We define the number of virtual qubits $V$. This is related 


# ## PennyLane Design


# ### Import PennyLane and Numpy


import pennylane as qml
import numpy as np


# ### Build the circuit with PennyLane
# The following cell builds a quantum circuit in the shape of a matrix product state tensor network and prints the result.
# <br>
# Block defines a variational quantum circuit that takes the position of tensors in the network.


def block(weights,wires):
    qml.CNOT(wires=wires)
    qml.RX(weights[0],wires=wires[0])
    qml.RY(weights[1],wires=wires[1])


dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev)
def circuit(template_weights):
    qml.MPS(wires=range(4),n_block_wires=2,block=block,n_params_block=2,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,2])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
print(qml.draw(circuit,expansion_strategy='device')(weights))


# ### Easy circuit editing
# 
# Using the `qml.MPS` template we can easily change the block type, depth, and size. For example:


# ### A different block:


def different_block(weights, wires):
    qml.CNOT(wires=wires)
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.Rot(weights[0],weights[1],weights[2],wires=wires[0])

dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev)
def circuit(template_weights):
    qml.MPS(wires=range(4),n_block_wires=2,block=different_block,n_params_block=3,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,3])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
print(qml.draw(circuit,expansion_strategy='device')(weights))


# ### A deeper block


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

@qml.qnode(dev)
def circuit(template_weights):
    qml.MPS(wires=range(4),n_block_wires=2,block=deep_block,n_params_block=5,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,5])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
print(qml.draw(circuit,expansion_strategy='device')(weights))


# ### A larger block


def large_block(weights, wires):
    qml.MultiControlledX(control_wires=wires[0:3],wires=wires[3])
    qml.RX(weights[0],wires=wires[0])
    qml.RX(weights[1],wires=wires[1])
    qml.RX(weights[2],wires=wires[2])
    qml.RX(weights[3],wires=wires[3])

dev= qml.device('default.qubit',wires=8)

@qml.qnode(dev)
def circuit(template_weights):
    qml.MPS(wires=range(8),n_block_wires=4,block=large_block,n_params_block=5,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=7))


weights = np.random.random(size=[3,5])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
print(qml.draw(circuit,expansion_strategy='device')(weights))


# ### A different tensor network
# We can also easily call a different tensor network architecture by using `qml.TTN`


def block(weights, wires):
    qml.CNOT(wires=wires)
    qml.RX(weights[0],wires=wires[0])
    qml.RX(weights[1],wires=wires[1])

dev= qml.device('default.qubit',wires=4)

@qml.qnode(dev)
def circuit(template_weights):
    qml.TTN(wires=range(4),n_block_wires=2,block=block,n_params_block=2,template_weights=template_weights)
    return qml.expval(qml.PauliZ(wires=3))


weights = np.random.random(size=[3,2])
print(f'The circuit result is {circuit(weights):.02f}')
print('The circuit looks like:')
print(qml.draw(circuit,expansion_strategy='device')(weights))


# ## 



"""
