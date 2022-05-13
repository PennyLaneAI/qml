r"""
.. _learning_few_data:

Generalization in quantum machine learning from few training data
==========================================

.. meta::
    :property="og:description": Generalization of quantum machine learning models.
    :property="og:image": https://pennylane.ai/qml/_images/few_data_thumbnail.png

.. related::

    tutorial_local_cost_functions Alleviating barren plateaus with local cost functions

*Authors: asdasd. Posted: 01 June 2022*

This demo is reproducing the results in `Generalization in quantum machine learning from few training data <https://arxiv.org/abs/2111.05292>`__ `[1] <#ref1>`__  
by Matthias Caro and co-authors. The authors find bounds on the necessary training data to guarantuee generalization for quantum machine learning (QML) tasks.
"""

##############################################################################
# Generalization Bounds for Quantum Machine Learning Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# | X
# | X
# | Some high level theoretical results here ...
# | X
# | X
# | some more detailed results (e.g. specifically for QCNNs etc. ...
# | X
# | X

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
# *Generating the training data*
# We are considering the transverse field Ising model Hamiltonian
#
# .. math:: H = -\sum_i \sigma_i^z \sigma_{i+1}^z - h\sum_i \sigma_i^x.
#
# We compute the ground state for 50 different values of the transverse
# field h. We use `L = 10` which is a trade off between finite size effects
# and simulation time. The phase transition in the thermodynamic limit
# :math:`L\rightarrow \infty` is known to be at `h=1`, but as we see below,
# for our finite size system we can estimate it around `h=0.5`. This is important
# for the classification later.
#

import pennylane as qml
import scipy.sparse.linalg as linalg
import numpy as np
from matplotlib import pyplot as plt


def H_ising(h, n_wires):
    ops = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n_wires - 1)]
    ops += [qml.PauliX(i) for i in range(n_wires)]
    ops += [qml.PauliZ(i) for i in range(n_wires)]  # extra term to break the symmetry
    coefs = [-1 for _ in range(n_wires - 1)]
    coefs += [-h for _ in range(n_wires)]
    coefs += [-1e-03 for _ in range(n_wires)]
    return qml.Hamiltonian(coefs, ops)


def ising(h, n_wires):
    H = H_ising(h, n_wires)
    Hmat = qml.utils.sparse_hamiltonian(H)
    E, V = linalg.eigsh(Hmat, k=1, which="SA", return_eigenvectors=True, ncv=20)
    return V[:, 0], E[0]


n_wires = 10
hs = np.linspace(0, 2, 50)
res = np.array([ising(h, n_wires) for h in hs], dtype=object)  # should be <10s
dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev)
def magz(vec):
    qml.QubitStateVector(vec, wires=range(n_wires))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]


mzs = np.array([magz(vec) for vec in res[:, 0]])
mzs = np.sum(mzs, axis=-1) / n_wires
plt.plot(hs, mzs, "x--")


##############################################################################
# We now take these ground states as our data for training and testing. We can
# initialize any circuit in PennyLane using `qml.QubitStateVector()` as shown
# in the example below.

data = res[:, 0]

L = 10
dev = qml.device("default.qubit", wires=L)


@qml.qnode(dev)
def example_circuit():
    qml.QubitStateVector(data[0], wires=range(L))
    return qml.state()


print(all(data[0] == example_circuit()))


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
    n_wires = len(wires)
    assert n_wires >= 3, "this circuit is too small!"

    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                if indx % 2 == 0 and skip_first_layer:
                    qml.U3(*weights[:3], wires=[w])
                    qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                qml.U3(*weights[9:12], wires=[w])
                qml.U3(*weights[12:], wires=[wires[indx + 1]])


def pooling_layer(weights, wires):
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"

    for indx, w in enumerate(wires):
        if indx % 2 == 1 and indx < n_wires:
            m_outcome = qml.measure(w)
            qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])


def conv_and_pooling(kernel_weights, n_wires):
    convolutional_layer(kernel_weights[:15], n_wires)
    pooling_layer(kernel_weights[15:], n_wires)


def dense_layer(weights, wires):
    qml.ArbitraryUnitary(weights, wires)


##############################################################################
# Let us now define a circuit using a ``qml.qnode`` that
# .

n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def circuit(weights, last_layer_weights, input_state):
    assert weights.shape[0] == 18, "The size of your weights vector is incorrect!"

    layers = weights.shape[1]
    wires = list(range(n_qubits))

    # inputs the state input_state
    qml.QubitStateVector(input_state, wires=wires)

    # adds convolutional and pooling layers
    for j in range(layers):
        conv_and_pooling(weights[:, j], wires)
        wires = wires[::2]

    assert (
        last_layer_weights.size == 4 ** (len(wires)) - 1
    ), f"The size of the last layer weights vector is incorrect! \n Expected {4**(len(wires)) - 1 }, Given {last_layer_weights.size}"
    dense_layer(last_layer_weights, wires)
    return qml.probs(wires=wires)


qml.draw_mpl(circuit)(np.random.rand(18, 3), np.random.rand(4**2 - 1), np.random.rand(2**16))

# .. figure:: ../demonstrations/learning_few_data/qcnn.png
#     :width: 100%
#     :align: center

##############################################################################
# Performance vs. training dataset size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can repeat the above analysis with increasing size of the training dataset.


qubits = [2, 3, 4, 5, 6]


######################################################################
# References
# ----------
#
# [1] *Generalization in quantum machine learning from few training data*, Matthias Caro
# et. al., `arxiv:2111.05292 <https://arxiv.org/abs/2111.05292>`__ (2021)
