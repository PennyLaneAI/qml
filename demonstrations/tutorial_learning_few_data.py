r"""
.. _learning_few_data:

Generalization in QML from few training data
==========================================

.. meta::
   :property="og:description": some description.
   :property="og:image": https://pennylane.ai/qml/_images/surface.png

.. related::

   tutorial_local_cost_functions Alleviating barren plateaus with local cost functions

*Author: XYZ (XYZ@gmail.com). Last updated: 26 Oct 2020.*

In some intro....

 e.g., QAOA (Quantum Adiabatic Optimization Algorithm)
which can be found in this `PennyLane QAOA tutorial
<https://pennylane.readthedocs.io/en/latest/tutorials/pennylane_run_qaoa_maxcut.html#qaoa-maxcut>`_.

somethingsomething

*"fsoomethinggggg."*

Thus, somethingsomething.


.. figure:: ../demonstrations/learning_few_data/qcnn.png
   :width: 90%
   :align: center
   :alt: surface

|

In this tutorial, we do something.

.. note::

    **tile**

    Some notes:

    *"Some note."*

SOME TITLE
---------------------------------------------------

First, we import PennyLane, NumPy, and Matplotlib
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


##############################################################################
# Next, we create a randomized variational circuit

# Set a seed for reproducibility
np.random.seed(42)

def rand_circuit(params, random_gate_sequence=None, num_qubits=None):
    pass


##############################################################################
# Now here we have some text
# ``gradient[-1]`` only.

grad_vals = []
num_samples = 200




##############################################################################
# QCNN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define QCNN.

def convolutional_layer(weights, wires, skip_first_layer=True): 
    n_wires= len(wires)
    assert n_wires>=3, "this circuit is too small!"

    for p in [0,1]:
        for indx,w in enumerate(wires):
            if indx%2==p and indx<n_wires-1:
                if indx%2==0 and skip_first_layer:
                    qml.U3(*weights[:3], wires = [w])
                    qml.U3(*weights[3:6], wires = [wires[indx+1]])
                qml.IsingXX(weights[6], wires = [w, wires[indx+1]])
                qml.IsingYY(weights[7], wires = [w, wires[indx+1]])
                qml.IsingZZ(weights[8], wires = [w, wires[indx+1]])
                qml.U3(*weights[9:12], wires = [w])
                qml.U3(*weights[12:], wires = [wires[indx+1]])

def pooling_layer(weights, wires):
    n_wires= len(wires)
    assert len(wires)>=2, "this circuit is too small!"
    
    for indx,w in enumerate(wires):
        if indx%2==1 and indx<n_wires:
            m_outcome = qml.measure(w)
            qml.cond(m_outcome, qml.U3)(*weights, wires = wires[indx-1])

def conv_and_pooling(kernel_weights, n_wires):
    convolutional_layer(kernel_weights[:15], n_wires)
    pooling_layer(kernel_weights[15:], n_wires)
    
def dense_layer(weights, wires):
    qml.ArbitraryUnitary(weights, wires)

##############################################################################
# Define circuit blah blah
# ``gradient[-1]`` only.

n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(weights, last_layer_weights):
    assert weights.shape[0]==18, "The size of your weights vector is incorrect!"
    
    layers = weights.shape[1]
    wires = list(range(n_qubits))
    
    for j in range(layers):
        conv_and_pooling(weights[:,j], wires)
        wires = wires[::2]
    
    assert last_layer_weights.size == 4**(len(wires)) - 1, f"The size of the last layer weights vector is incorrect! \n Expected {4**(len(wires)) - 1 }, Given {last_layer_weights.size}"
    dense_layer(last_layer_weights, wires)
    return qml.probs(wires=wires)

qml.draw_mpl(circuit)(np.random.rand(18,3), np.random.rand(4**2 -1))

##############################################################################
# Performance vs. training dataset size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can repeat the above analysis with increasing size of the training dataset.


qubits = [2, 3, 4, 5, 6]



##############################################################################
# References
# ----------
#
# 1. Dauphin, Yann N., et al.,
#    Identifying and attacking the saddle point problem in high-dimensional non-convex
#    optimization. Advances in Neural Information Processing
#    systems (2014).
#
# 2. McClean, Jarrod R., et al.,
#    Barren plateaus in quantum neural network training landscapes.
#    Nature communications 9.1 (2018): 4812.
#
# 3. Grant, Edward, et al.
#    An initialization strategy for addressing barren plateaus in
#    parametrized quantum circuits. arXiv preprint arXiv:1903.05076 (2019).
