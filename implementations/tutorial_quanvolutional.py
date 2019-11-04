r"""
.. _quanvolutional:

Quanvolutional Neural Networks
==============================
*Author: Andrea Mari*

In this tutorial we implement the *Quanvolutional Neural Network*, a quantum
machine learning model originally introduced in
`Henderson et al. (2019) <https://arxiv.org/abs/1904.04767>`_.



Introduction
------------

General setup
------------------------
This Python code requires *PennyLane* and the plotting library *matplotlib*.
"""

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import matplotlib.pyplot as plt

##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_qubits = 3    # Number of system qubits.
steps = 30      # Number of optimization steps
eta = 0.8       # Learning rate
rng_seed = 0    # Seed for random number generator



##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. 
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
# 
