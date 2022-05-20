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
# | Some high level theoretical results here ... AND I JUST MADE A CHANGE
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
import pennylane.numpy as pnp
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


n_wires = 8
hs = np.linspace(0, 2, 200)
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

n_wires = 8
dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev)
def conv_net(weights, last_layer_weights, input_state):
    assert weights.shape[0] == 18, "The size of your weights vector is incorrect!"

    layers = weights.shape[1]
    wires = list(range(n_wires))

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
    return qml.probs(wires=(0))


qml.draw_mpl(conv_net)(np.random.rand(18, 2), np.random.rand(4**2 - 1), np.random.rand(2**16))

# .. figure:: ../demonstrations/learning_few_data/qcnn.png
#     :width: 100%
#     :align: center

##############################################################################
# Performance vs. training dataset size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can repeat the above analysis with increasing size of the training dataset.
# First, we create the labels and draw ``N_train`` random training samples. Additionally, we also draw
# ``N_val`` validation samples that acts as a proxy for the full test set during training.

labels = np.zeros(len(data), dtype=int)
labels[np.where(mzs<0.5)] = 1
N_train = 20
N_val = 20
randomstate = np.random.default_rng( 0 )
choice = randomstate.choice(len(data), N_train + N_val, replace=False, shuffle=False )

train_choice = choice[:N_train]
val_choice = choice[N_train:]
train_data = data[train_choice]
train_labels = labels[train_choice]
train_hs = hs[train_choice]

val_data = data[val_choice]
val_labels = labels[val_choice]
val_hs = hs[val_choice]

##############################################################################
# We now define the loss function that we want to optimize. In this case we only have two classes, which allows us to use the output
# of one qubit as the label (here: the first). More specifically, we use the probability of measuring ``0`` as the label for the ferromagnetic phase, and
# the probability of measuring ``1`` as the label for the paramagnetic phase. Mathematically, the cost function that we
# are then trying to optimize is 
#
# .. math:: \mathcal{L} = \sum_i p(y_i)
#
# where :math:`y_i \in \{0, 1\}` is the corresponding label of training example `i`.
#
# In our implementation in PennyLane, we simply achieve this by taking the corresponding entry of the ``qml.probs`` output for the first qubit:

def loss_fn(weights, weights_last, data, labels):
    return 1-qml.math.sum([conv_net(weights, weights_last, state)[label] for state, label in zip(data, labels)])/len(data)

##############################################################################
# Similarily, we compute the accuracy, which is the relative frequency of guessing the right label, i.e. :math:`p(y_i)>0.5`.

def accuracy(weights, weights_last, data, labels):
    return qml.math.sum([conv_net(weights, weights_last, state)[label]>0.5 for state, label in zip(data, labels)])/len(data)

##############################################################################
# In order to use PennyLane's automatid differentiation we specify the trainable parameters, which will later then be automatically picked up

weights = pnp.random.rand(18, 2, requires_grad=True)
weights_last = pnp.random.rand(4**(2)-1, requires_grad = True)

##############################################################################
# We can now train

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
train_loss = [] ; val_loss = []
train_acc = [] ; val_acc = []

n_iter = 40
for k in range(n_iter):
    if k % 10 == 0:
        print(f"Step {k+1} / {n_iter}, cost: {old_loss}")
        train_acc.append(accuracy(weights, weights_last, train_data, train_labels))
        val_acc.append(accuracy(weights, weights_last, val_data, val_labels))
        print(f"train_acc = {train_acc[-1]} val_acc = {val_acc[-1]}")

    (weights, weights_last), old_loss = optimizer.step_and_cost(loss_fn, weights, weights_last, data=train_data, labels=train_labels)
    train_loss.append(old_loss)
    val_loss.append(loss_fn(weights, weights_last, val_data, val_labels))

train_loss.append(loss_fn(weights, weights_last, train_data, train_labels))
val_loss.append(loss_fn(weights, weights_last, val_data, val_labels))

##############################################################################
# We can check if the training was successful by looking at the loss and accuracy for the training and validation set during training:

fig, axs = plt.subplots(ncols=2, figsize=(10,5))
ax = axs[0]
ax.plot(losses,"x--")

ax = axs[1]
ax.plot(np.arange(0,n_iter,10), val_acc,"x--", label="val")
ax.plot(np.arange(0,n_iter,10), train_acc,"o:", label="train")
ax.legend(fontsize=20)
ax.set_ylabel("accuracy", fontsize=20)
ax.set_xlabel("epoch", fontsize=20)

plt.tight_layout()
plt.show()

##############################################################################
# We can also look at the phase diagram directly and see for which transverse field parameters the training is successful:

out = [np.argmax(conv_net(weights, weights_last, state)) for state in data]
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(hs, out,".--", label="class")
ax.plot(hs, mzs,"o:", label="mag")
ax.legend(fontsize=20)
ax.set_xlabel("h", fontsize=20)
plt.show()


######################################################################
# References
# ----------
#
# [1] *Generalization in quantum machine learning from few training data*, Matthias Caro
# et. al., `arxiv:2111.05292 <https://arxiv.org/abs/2111.05292>`__ (2021)
