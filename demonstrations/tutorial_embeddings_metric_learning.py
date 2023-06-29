"""
.. _embeddings_metric_learning:

.. role:: html(raw)
   :format: html

Quantum embeddings and metric learning
======================================

.. meta::
    :property="og:description": Train a quantum embedding to encode data from the same classes
        close together as quantum states.
    :property="og:image": https://pennylane.ai/qml/_images/training.png

*Authors: Maria Schuld and Aroosa Ijaz — Posted: 14 January 2020. Last updated: 01 July 2023.*

Metric learning is a paradigm in supervised machine learning that aims at decreasing 
the distance between representations of training examples from the same class while increasing 
the distance between those from different classes. After the representation is learned, a simple 
classifier can distinguish between the classes. 

The approach of metric learning has been investigated 
from the perspective of quantum computing in `Lloyd, Schuld, Ijaz, Izaac, Killoran (2019) <https://arxiv.org/abs/2001.03622>`_.
This demo reproduces some results from the paper
by training a hybrid classical-quantum data
embedding to classify images of ants and bees (inspired by this `tutorial <https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html>`_). 
"""


######################################################################
# The tutorial requires the following imports:
#

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import optax
import pennylane as qml
import numpy as np
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)
from sklearn.preprocessing import StandardScaler

######################################################################
# Idea
# ----
#
# Quantum metric learning trains a quantum embedding—a
# quantum circuit that encodes classical data into quantum states—to
# separate different classes of data in the Hilbert space of the quantum
# system.
#
# .. figure:: ../demonstrations/embedding_metric_learning/training.png
#    :align: center
#    :width: 40%
#
# The trained embedding can be used for classification. A new data sample
# (red dot) gets mapped into Hilbert space via the same embedding, and a
# special measurement compares it to examples from the two embedded classes.
# Note that the decision boundary of the measurement in quantum state space is nearly
# linear (red dashed line).
#
# .. figure:: ../demonstrations/embedding_metric_learning/classification.png
#    :align: center
#    :width: 40%
#
# Since a simple metric in Hilbert space corresponds to a potentially much
# more complex metric in the original data space, the simple decision
# boundary can translate to a non-trivial decision boundary in the
# original space of the data.
#
# .. figure:: ../demonstrations/embedding_metric_learning/dec_boundary.png
#    :align: center
#    :width: 40%
#
# The best quantum measurement one could construct to classify new inputs
# depends on the loss defined for the classification task, as well as on the
# metric used to optimize the separation of data.
#
# For a linear cost function, data separated by the trace distance or
# :math:`\ell_1` metric is best distinguished by a Helstrom measurement, while
# data separated by the Hilbert-Schmidt distance or :math:`\ell_2` metric
# is best classified by a fidelity measurement. Here we show how to
# implement training and classification based on the :math:`\ell_2`
# metric.
#
# Embedding
# ---------
#
# A quantum embedding is a representation of data points :math:`x` from a
# data domain :math:`X` as a *(quantum) feature state*
# :math:`| x \rangle`. Either the full embedding, or part of it, can be
# facilitated by a "quantum feature map", a quantum circuit
# :math:`\Phi(x)` that depends on the input. If the circuit has additional
# parameters :math:`\theta` that are adaptable,
# :math:`\Phi = \Phi(x, \theta)`, the quantum feature map can be trained
# via optimization.
#
# In this tutorial we investigate a trainable, hybrid classical-quantum embedding
# implemented by a partially pre-trained classical neural network,
# followed by a parametrized quantum circuit that implements the quantum
# feature map:
#
# |
#
# .. figure:: ../demonstrations/embedding_metric_learning/pipeline.png
#    :align: center
#    :width: 100%
#
# |
#
# Following `Mari et al. (2019) <https://arxiv.org/abs/1912.08278>`__,
# for the classical neural network we use PyTorch's
# ``torch.models.resnet18()``, setting ``pretrained=True``. The final
# layer of the ResNet, which usually maps a 512-dimensional vector to 1000
# nodes representing different image classes, is replaced by a linear
# layer of 2 output neurons. The classical part of the embedding therefore
# maps the images to a 2-dimensional *intermediate feature space*.
#
# For the quantum part we use the QAOA embedding proposed
# in `Lloyd et al. (2019) <https://arxiv.org/abs/2001.03622>`_.
# The feature map is represented by a layered variational circuit, which
# alternates a "feature-encoding Hamiltonian" and an "Ising-like" Hamiltonian
# with ZZ-entanglers (the two-qubit gates in the circuit diagram above) and ``RY`` gates as local fields.
#


def feature_encoding_hamiltonian(features, wires):
    for idx, w in enumerate(wires):
        qml.RX(features[idx], wires=w)


def ising_hamiltonian(weights, wires, l):
    # ZZ coupling
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(weights[l, 0], wires=wires[0])
    qml.CNOT(wires=[wires[1], wires[0]])
    # local fields
    for idx, w in enumerate(wires):
        qml.RY(weights[l, idx + 1], wires=w)


def QAOAEmbedding(features, weights, wires):
    repeat = len(weights)
    for l in range(repeat):
        # apply alternating Hamiltonians
        feature_encoding_hamiltonian(features, wires)
        ising_hamiltonian(weights, wires, l)
    # repeat the feature encoding once more at the end
    feature_encoding_hamiltonian(features, wires)


######################################################################
# .. note:: Instead of using the hand-coded ``QAOAEmbedding()`` function, PennyLane provides
#           a built-in :func:`QAOAEmebedding <pennylane.templates.QAOAEmbedding>` template.
#           To use it, simply replace the cell above
#           by ``from pennylane.templates import QAOAEmbedding``. This will also allow you to use
#           a different number of qubits in your experiment.
#
# Overall, the embedding has 1024 + 12 trainable parameters - 1024 for the
# classical part of the model and 12 for the four layers of the QAOA
# embedding.
#
# .. note:: The pretrained neural network has already learned
#           to separate the data. The example does therefore not
#           make any claims on the performance of the embedding, but aims to
#           illustrate how a hybrid embedding can be trained.
#
# Data
# ----
#
# We consider a binary supervised learning problem with examples
# :math:`\{a_1,...a_{M_a}\} \subseteq X` from class :math:`A` and examples
# :math:`\{b_1,...b_{M_b}\} \subseteq X` from class :math:`B`. The data
# are images of ants (:math:`A`) and bees (:math:`B`), taken from `Kaggle's
# hymenoptera dataset <https://www.kaggle.com/ajayrana/hymenoptera-data>`__.
# This is a sample of four images:
#
# .. figure:: ../demonstrations/embedding_metric_learning/data_example.png
#    :align: center
#    :width: 50%
#
# For convenience, instead of coding up the classical neural network, we
# load `pre-extracted feature vectors of the images
# <https://github.com/XanaduAI/qml/blob/master/demonstrations/embedding_metric_learning/X_antbees.txt>`_.
# These were created by
# resizing, cropping and normalizing the images, and passing them through
# PyTorch's pretrained ResNet 512 (that is, without the final linear layer)
# (see `script used for pre-processing
# <https://github.com/XanaduAI/qml/blob/master/demonstrations/embedding_metric_learning/image_to_resnet_output.py>`_).
#

X = np.loadtxt("embedding_metric_learning/X_antbees.txt")  # pre-extracted inputs
Y = np.loadtxt("embedding_metric_learning/Y_antbees.txt")  # labels
X_val = np.loadtxt(
    "embedding_metric_learning/X_antbees_test.txt"
)  # pre-extracted validation inputs
Y_val = np.loadtxt("embedding_metric_learning/Y_antbees_test.txt")  # validation labels
Y[Y == 0] = -1  # rename label 0 to -1
Y_val[Y_val == 0] = -1

# split data into two classes
A = X[Y == -1]
B = X[Y == 1]
A_val = X_val[Y_val == -1]
B_val = X_val[Y_val == 1]

print(A.shape)
print(B.shape)


######################################################################
# Cost
# ----
#
# The distance metric underlying the notion of 'separation' is the
# :math:`\ell_2` or Hilbert-Schmidt norm, which depends on overlaps of
# the embedded data points :math:`|a\rangle`
# from class :math:`A` and :math:`|b\rangle` from class :math:`B`,
#
# .. math::
#
#     D_{\mathrm{hs}}(A, B) =  \frac{1}{2} \big( \sum_{i, i'} |\langle a_i|a_{i'}\rangle|^2
#        +  \sum_{j,j'} |\langle b_j|b_{j'}\rangle|^2 \big)
#        - \sum_{i,j} |\langle a_i|b_j\rangle|^2.
#
# To maximize the :math:`\ell_2` distance between the two classes in
# Hilbert space, we minimize the cost
# :math:`C = 1 - \frac{1}{2}D_{\mathrm{hs}}(A, B)`.
#
# To set up the "quantum part" of the cost function in PennyLane, we have
# to create a quantum node. Here, the quantum node is simulated on
# PennyLane's ``'default.qubit'`` backend.
#
# .. note:: One could also connect the
#           quantum node to a hardware backend to find out if the noise of a
#           physical implementation still allows us to train the embedding.
#

n_features = 2
n_qubits = 2 * n_features + 1

dev = qml.device("default.qubit", wires=n_qubits)


######################################################################
# We use a the following circuit to measure the overlap
# :math:`|\langle \psi | \phi \rangle|^2` between two quantum feature
# states :math:`|\psi\rangle` and :math:`|\phi\rangle`, prepared by a
# ``QAOAEmbedding`` with weights ``q_weights``:
#


@jax.jit
@qml.qnode(qml.device("default.qubit", wires=2))
def overlap(q_weights, x1, x2):
    qml.QAOAEmbedding(features=x1, weights=q_weights, wires=[0, 1])
    qml.adjoint(qml.QAOAEmbedding)(features=x2, weights=q_weights, wires=[0, 1])

    return qml.expval(qml.Projector(np.array([0, 0]), wires=[0, 1]))


######################################################################
# Note that we could have used a swap test, but the above circuit has
# the same result while using fewer resources (for more details check
# `this tutorial <https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html>`_).
# The circuit above is marked by the `@jax.jit` decorator so that it gets
# compiled during the first call. Metric learning is very circuit
# hungry, and this trick speeds training and prediction up significantly.
#

######################################################################
# Before executing the circuit, the feature vectors have to be
# multiplied by a (2, 512)-dimensional matrix that represents the weights
# of the linear layer. This trainable classical pre-processing is executed
# before calling the circuit:
#


def overlaps(params, X1=None, X2=None):
    res = 0
    for x1 in X1:
        for x2 in X2:
            x1_feats = params["params_classical"] @ x1
            x2_feats = params["params_classical"] @ x2
            res += overlap(params["params_quantum"], x1_feats, x2_feats)
    return res / (len(X1) * len(X2))


######################################################################
# In the ``overlaps()`` function, ``params`` is a dictionary of two arrays --
# the matrix of the linear layer and the quantum circuit parameters --
# which we define below.
#
# With this we can define the cost function :math:`C`, which depends on
# inter- and intra-cluster overlaps.
#


def cost_fn(params, A=None, B=None):
    aa = overlaps(params, X1=A, X2=A)
    bb = overlaps(params, X1=B, X2=B)
    ab = overlaps(params, X1=A, X2=B)

    d_hs = -ab + 0.5 * (aa + bb)
    return 1 - d_hs


######################################################################
# Optimization
# ------------
# The initial parameters for the trainable classical and quantum part of the embedding are
# chosen at random. The number of layers in the quantum circuit is derived from the first
# dimension of `init_pars_quantum`.
#

# generate initial parameters for circuit (4 layers)
init_pars_quantum = jnp.array(np.random.normal(loc=0, scale=0.1, size=(4, 3)))

# generate initial parameters for linear layer
init_pars_classical = jnp.array(np.random.normal(loc=0, scale=0.1, size=(2, 512)))

params = {"params_quantum": init_pars_quantum, "params_classical": init_pars_classical}

######################################################################
# Since training and estimating the progress on the entire dataset
# is very costly due to the pairwise comparisons, the following
# function to randomly subsample data from each class will be handy:
#


def get_batch(batch_size, A, B):
    selectA = np.random.choice(range(len(A)), size=(batch_size,), replace=False)
    selectB = np.random.choice(range(len(B)), size=(batch_size,), replace=False)
    return A[selectA], B[selectB]


######################################################################
# We can now train the embedding with an Adam optimizer, sampling
# 4 training points from each class in every step. This means that
# in every evaluation of the cost, `3*4^2 = 48` overlaps are calculated
# by the quantum device.
#
# To monitor training we sample 20 data points from the training and validation set
# and compute the loss on these samples only. One needs to keep in mind that 
# the loss function is different from standard machine learning since it measures 
# state overlaps. For example, the training cost will not converge to zero, 
# since this would mean that all data from a class maps to the same quantum state. 
# Furthermore, the cost fluctuates strongly during training since the batch size of 
# data used for computing gradients and for monitoring progress is small. 
#


@jax.jit
def update(params, opt_state, A, B):
    loss, grads = jax.value_and_grad(cost_fn)(params, A, B)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)
batch_size = 4

A_monitor, B_monitor = get_batch(20, A, B)
A_monitor_val, B_monitor_val = get_batch(20, A_val, B_val)

for i in range(100000):
    A_batch, B_batch = get_batch(batch_size, A, B)
    params, opt_state, loss = update(params, opt_state, A_batch, B_batch)

    if i % 1000 == 0:
        # monitor progress on the 20 validation samples
        test_loss = cost_fn(params, A_monitor, B_monitor)
        train_loss = cost_fn(params, A_monitor_val, B_monitor_val)

        print(f"Step {i} Estimated loss train: {train_loss} | test: {test_loss}")


######################################################################
# Analysis
# --------
#
# Let us analyze the effect of training.
#

######################################################################
# A useful way to visualize the distance of data points is to plot a Gram
# matrix of the overlaps of different feature states. For this we use the
# 20 data samples employed to monitor training above.
#
# After training, the gram matrix clearly separates the two classes
# on the training set.
#

A_B_train = np.r_[A_monitor, B_monitor]

gram = [[overlaps(params, X1=[x1], X2=[x2]) for x1 in A_B_train] for x2 in A_B_train]
ax = plt.subplot(111)
im = ax.matshow(gram, vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

######################################################################
# The gram matrix also somewhat separates the two classes
# on the validation set.
#

A_B_val = np.r_[A_monitor_val, B_monitor_val]

gram = [[overlaps(params, X1=[x1], X2=[x2]) for x1 in A_B_val] for x2 in A_B_val]
ax = plt.subplot(111)
im = ax.matshow(gram, vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

######################################################################
# We can also visualize the "intermediate layer" of 2-dimensional vectors
# :math:`(x_1, x_2)`, just before feeding them into the quantum circuit.
# After training, the linear layer learned to arrange the
# intermediate feature vectors.
#

A_2d = (params["params_classical"] @ A.T).T
plt.scatter(A_2d[:, 0], A_2d[:, 1], c="red", s=5)

B_2d = (params["params_classical"] @ B.T).T
plt.scatter(B_2d[:, 0], B_2d[:, 1], c="blue", s=5)

A_val_2d = (params["params_classical"] @ A_val.T).T
plt.scatter(A_val_2d[:, 0], A_val_2d[:, 1], c="orange", s=10)

B_val_2d = (params["params_classical"] @ B_val.T).T
plt.scatter(B_val_2d[:, 0], B_val_2d[:, 1], c="green", s=10)

plt.show()


######################################################################
# Classification
# --------------
#
# Given a new input :math:`x \in X`, and its quantum feature state
# :math:`|x \rangle`, the trained embedding can be used to solve the
# binary classification problem of assigning :math:`x` to either :math:`A`
# or :math:`B`. For an embedding separating data via the :math:`\ell_2`
# metric, a very simple measurement can be used for classification: one
# computes the overlap of :math:`|x \rangle` with examples of
# :math:`|a \rangle` and :math:`|b \rangle`. :math:`x` is assigned to the
# class with which it has a larger average overlap in the space of the
# embedding.
#
# Let us consider a picture of an ant from the validation set (assuming
# our model never saw it during training):
#
# |
#
# .. figure:: ../demonstrations/embedding_metric_learning/ant.jpg
#    :align: center
#    :width: 40%
#
# |
#
# After passing it through the classical neural network (excluding the final
# linear layer), the 512-dimensional feature vector is given by
# ``A_val[0]``.

x_new = A_val[0]

print(x_new.shape)


######################################################################
# We compare the new input with randomly selected samples. The more
# samples used, the smaller the variance in the prediction.
#

n_samples = 200

prediction = 0
for s in range(n_samples):
    # select a random sample from the training set
    sample_index = np.random.choice(len(X))
    x = X[sample_index]
    y = Y[sample_index]

    # compute the overlap between training sample and new input
    overlap = overlaps(params, X1=[x], X2=[x_new])

    # add the label weighed by the overlap to the prediction
    prediction += y * overlap

# normalize prediction
prediction = prediction / n_samples
print(prediction)


######################################################################
# Since the result is negative, the new data point is (correctly) predicted
# to be a picture of an ant, which was the class with -1 labels.
#
# References
# ----------
# Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran: "Quantum embeddings for machine learning"
# arXiv preprint arXiv:2001.03622.
#
# Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, Nathan Killoran: "Transfer learning
# in hybrid classical-quantum neural networks" arXiv preprint arXiv:1912.08278
#

##############################################################################
# About the authors
# -----------------
#
# .. include:: ../_static/authors/maria_schuld.txt
# .. include:: ../_static/authors/aroosa_ijaz.txt
