r"""
.. _learning_few_data:

Generalization in quantum machine learning from few training data
==========================================

.. meta::
    :property="og:description": Generalization of quantum machine learning models.
    :property="og:image": https://pennylane.ai/qml/_images/few_data_thumbnail.png

.. related::

    tutorial_local_cost_functions Alleviating barren plateaus with local cost functions

*Authors: Korbinian Kottmann, Luis Mantilla Calderon, Maurice Weber. Posted: 01 June 2022*

In this tutorial we dive into the generalization capabilities of quantum machine learning models.
For the example of a Quantum Convolutional Neural Nework (QCNN), we show how its generalization error behaves as a
function of the number of training samples. This demo is based on the paper
*"Generalization in quantum machine learning from few training data"*. by Caro et al. [#CaroGeneralization]_.

What is Generalization in (Q)ML?
------------------------
When optimizing a machine learning model, be it classical or quantum, we aim to maximize its performance over the data
distribution of interest, like images of cats and dogs. However, in practice we are limited to a finite amount of
data, necessitating the need to reason about how our model performs on new, previously unseen data. The difference
between the model's performance on the true data distribution, and the performance estimated from our training data is
called the *generalization error* and indicates how well the model has learned to generalize to unseen data. It is good
to know that generalization can be seen as a manifestation of the bias-variance trade-off: models which
perfectly fit the training data, i.e. which admit a low bias, have a higher variance, typically perform poorly on unseen
test data and don't generalize well. In the classical machine learning community, this trade off has been extensively
studied and has lead to optimization techniques which favour generalization, for example by regularizing models via
their variance [#NamkoongVariance]_.

Let us now dive deeper into generalization properties of quantum machine learning (QML) models. We start by describing
the typical data processing pipeline of a QML model. A classical data input :math:`x` is first encoded in a quantum
state via a mapping :math:`x \mapsto \rho(x)`. This encoded state is then processed through a parametrized quantum
channel :math:`\rho(x) \mapsto \mathcal{E}_\alpha(\rho(x))` and a measurement is performed on the resulting state
to get the final prediction. The goal is now to minimize the expected loss over the data generating distribution
:math:`P` indicating how well our model performs on new data. Mathematically, for a loss function :math: `\ell`, the
expected loss is given by

.. math:: R(\alpha) = \mathbb{E}_{(x,y)\sim P}[\ell(\alpha;\,x,\,y)].

As :math:`P` is generally not known, in practice this quantity has to be estimated from a finite amount of data. Given
a training set :math:`S = \{(x_i,\,y_i)\}_{i=1}^N`, we estimate the performance of our QML model by calculating the
average loss over the training set

.. math:: \hat{R}_S(\alpha) = \frac{1}{N}\sum_{i=1}^N \ell(\alpha;\,x_i,\,y_i)

which is referred to as the training loss and is an unbiased estimate of :math:`R(\alpha)`. This is only a proxy
to the true quantity of interest :math:`R(\alpha)` and their difference is called the generalization error

.. math:: \mathrm{gen}(\alpha) = \hat{R}_S(\alpha) - R(\alpha)

which is the quantity that we explore in this tutorial. Keeping in mind the bias-variance trade off, one would expect
that more complex models, i.e. models with a larger number of parameters, achieve a lower error on the training data,
but a higher generalization error. Having more training data on the other hand leads to a better approximation of the
true expected loss and hence lower generalization error. This intuition is made precise in Ref. [#CaroGeneralization]_
where it is shown that :math:`\mathrm{gen}(\alpha)` roughly scales as :math:`\mathcal{O}(\sqrt{T / N})` where :math:`T`
is the number of parametrized gates and :math:`N` is the number of training samples.
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
import seaborn as sns
sns.set()

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
    return V[:, 0]


n_wires = 8
hs = np.linspace(0, 2, 200)
data = np.array([ising(h, n_wires) for h in hs])
dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev)
def magz(vec):
    qml.QubitStateVector(vec, wires=range(n_wires))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]


mzs = np.array([magz(vec) for vec in data])
mzs = np.sum(mzs, axis=-1) / n_wires
plt.plot(hs, mzs, "x--")
plt.xlabel("h", fontsize=20)
plt.ylabel("$\\langle \sigma_z \\rangle$", fontsize=20)
plt.show()


##############################################################################
# We now take these ground states as our data for training and testing. We can
# initialize any circuit in PennyLane using ``qml.QubitStateVector()`` as shown
# in the example below.

##############################################################################
# Quantum Convolutional Neural Netwokr
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let us now create a quantum CNN like the one  proposed by Cong, et al. 
# [#CongQuantumCNN]. Similar to a classical CNN, we have both a 
# `convolutional_layer` and a `pooling_layer`. The former layer acts as a window
# that extracts local correlations, while the former allows reducing the 
# dimensionality of the feature vector. In the simplest case, the 
# `convolutional_layer` consists of a two-qubit unitary that is shifted along 
# the circuit and the `pooling_layer` of a single qubit gate conditioned on the 
# measurement of a neighbouring qubit. These two layers are alternatingly 
# concatenated (conv-pool-conv-pool). Additionally, similar to classical CNNs, 
# we concatenate the reduced feature vector with a `dense layer`, which in our 
# case can be modeled as an all-to-all unitary gate.

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
# Let us now define a circuit that takes as an input the weights of the QCNN and
# the quantum state to be processed. We first take the vector representation of 
# the states calculated in `ising` and input them to a quantum circuit using
# ``qml.QuibtStateVector``. Then we use ``conv_and_pooling`` layers, followed by a
# ``dense_layer``. Finally, we calculate the probabilities of the outcomes ``{00, 01
# 10, 11}`` with ``qml.probs`` which will allow us to do the phase classification.

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
N_train = 10
N_val = 50

randomstate = np.random.default_rng( 0 )
safe_range = np.concatenate([np.where(mzs>0.9)[0], np.where(mzs<0.1)[0]]) # making sure to be away from the phase transition ~h in [0.5-1]
train_choice = randomstate.choice(safe_range, N_train, replace=False, shuffle=False )
val_choice = randomstate.choice(safe_range, N_val, replace=False, shuffle=False )

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

optimizer = qml.GradientDescentOptimizer(stepsize=0.01)
train_loss = [] ; val_loss = []
train_acc = [] ; val_acc = []

n_iter = 50
for k in range(0, n_iter):
    if not k == 0:
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

fig, axs = plt.subplots(ncols=3, figsize=(14,5))
ax = axs[0]
ax.plot(train_loss,"x--", label="train")
ax.plot(val_loss,"x--", label="val")
ax.set_ylabel("loss", fontsize=20)
ax.set_xlabel("epoch", fontsize=20)
ax.legend(fontsize=20)

ax = axs[1]
ax.plot(train_acc,"o:", label="train")
ax.plot(val_acc,"x--", label="val")
ax.set_ylabel("accuracy", fontsize=20)
ax.set_xlabel("epoch", fontsize=20)
ax.legend(fontsize=20)

ax = axs[2]
ax.plot(train_acc, val_acc,"o:")
ax.set_ylabel("val accuracy", fontsize=20)
ax.set_xlabel("train accuracy", fontsize=20)
ax.legend(fontsize=20)

plt.tight_layout()
#plt.savefig("few-data_loss_accuracy.png")
plt.show()

##############################################################################
# We can also look at the phase diagram directly and see for which transverse field parameters the training is successful:

out = [conv_net(weights, weights_last, state)[0] for state in data]
labels_predicted = [(0 if x>0.5 else 1) for x in out]


fig, ax = plt.subplots(figsize=(6,5))
ax.plot(hs, labels_predicted,".--", label="pred. class")
ax.plot(hs, labels, "o", label="actual class")
ax.plot(hs, mzs,"o:", label="mag")
ax.plot(hs, out, ".:", label="p(0)")
ax.legend(fontsize=20)
ax.set_xlabel("h", fontsize=20)
#plt.savefig("few-data_classification-result.png")
plt.show()


##############################################################################
# References
# ----------
#
# .. [#CaroGeneralization]
#
#     Matthias C. Caro, Hsin-Yuan Huang, M. Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, Patrick J. Coles.
#     "Generalization in quantum machine learning from few training data"
#     `arxiv:2111.05292 <https://arxiv.org/abs/2111.05292>`__, 2021.
#
# .. [#NamkoongVariance]
#
#     Hongseok Namkoong and John C. Duchi.
#     "Variance-based regularization with convex objectives."
#     `Advances in Neural Information Processing Systems
#     <https://proceedings.neurips.cc/paper/2017/file/5a142a55461d5fef016acfb927fee0bd-Paper.pdf>`__, 2017.
#
# .. [#CongQuantumCNN]
#
#     Iris Cong, Soonwon Choi, Mikhail D. Lukin.
#     "Quantum Convolutional Neural Networks"
#     `arxiv:1810.03787 <https://arxiv.org/abs/1810.03787>`__, 2018.
#