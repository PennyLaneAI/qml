"""
.. _kernel_based_training:

.. role:: html(raw)
   :format: html

Kernel-based training with scikit-learn
=======================================

.. meta:: 
    :property=“og:description”: Kernel-based training with scikit-learn. 
    :property=“og:image”: https://pennylane.ai/qml/_images/kernel_based_scaling.png

.. related::

    tutorial_variational_classifier Variational classifier
    
This demonstration illustrates how one can train quantum machine
learning models with a kernel-based approach instead of the usual
`variational
approach <https://pennylane.ai/qml/glossary/variational_circuit.html>`__.
The theoretical background has been established in many papers in the literature 
such as `Schuld and Killoran (2018) <https://arxiv.org/abs/1803.07128>`__, 
`Havlicek et al. (2018) <https://arxiv.org/abs/1804.11326>`__, 
`Liu (2020) <https://arxiv.org/abs/2010.02174>`__, `Huang et al. (2020) <https://arxiv.org/pdf/2011.01938.pdf>`__,
and has been summarised in the overview Schuld (2021) <https://arxiv.org/abs/2101.11020>`__ which we follow here.

As an example of kernel-based training we use a combination of PennyLane
and the powerful `scikit-learn <https://scikit-learn.org/>`__ machine
learning library to show how a support vector machine can be combined
with a quantum kernel. We then compare this strategy with a variational
quantum circuit trained via stochastic gradient descent and using
PyTorch.

The goal of the demo is to estimate the circuit evaluations needed in
both approaches. We will see that while kernel-based training has a much
worse scaling for big data sets, in the small-data regime of near-term
quantum computing it is much more efficient than variational training,
but becomes prohibitive for bigger datasets.

.. figure::  ../demonstrations/kernel_based_training/scaling.png 
       :align: center
       :scale: 20%
       :alt: Scaling of kernel-based vs. variational learning
       
"""

######################################################################
# Background
# ==========
#
# The main practical consequence of approaching quantum machine learning with a 
# kernel approach is that instead of training a quantum machine learning
# model of the form
#
# .. math:: f(x) = \langle \phi(x) | \mathcal{M} | \phi(x)\rangle
#
# we can often train a classical kernel method with a kernel executed on a
# quantum device. The “quantum” kernel
# is given by the mutual overlap of two data-encoding quantum states,
#
# .. math::  \kappa(x, x') = | \langle \phi(x') | \phi(x)\rangle|^2.
#
# Here, :math:`| \phi(x)\rangle` is a data-encoding quantum state prepared
# by a fixed `embedding
# circuit <https://pennylane.readthedocs.io/en/stable/introduction/templates.html#intro-ref-temp-emb>`__,
# and :math:`\mathcal{M}` an arbitrary observable. The observable can
# effectively be implemented by a simple measurement that is preceded by a
# quantum circuit. If the circuit is trainable, the measurement becomes
# trainable. For example, applying a circuit :math:`B(\theta)` and then
# measuring the PauliZ observable :math:`\sigma^0_z` of the first qubit
# implements the effective measurement observable
# :math:`\mathcal{M}(\theta) = B^{\dagger}(\theta) \sigma^0_z B(\theta)`.
#
# .. figure:: ../demonstrations/kernel_based_training/quantum_model.png 
#       :align: center
#       :scale: 20%
#       :alt: quantum-model
#
# Kernel-based training therefore by-passes the variational part and
# measurement of common variational circuits, and only depends on the
# embedding.
#
# .. note::
#
#    More precisely, we can replace variational training with kernel-based training if the optimisation
#    problem can be written as minimising a cost of the form
#    .. math:: f_{\rm trained} = \min_f  \lambda \mathrm{tr}\{\mathcal{M^2\} + \frac{1}{M}\sum_{m=1}^M L(f(x^m), y^m), 
#    which is a regularised empirical risk of training data samples :math:`(x^m, y^m)_{m=1\dots M}` and loss function :math:`L`.
#
# If the loss function in training is the `hinge
# loss <https://en.wikipedia.org/wiki/Hinge_loss>`__ the kernel method
# corresponds to a standard `support vector
# machine <https://en.wikipedia.org/wiki/Support-vector_machine>`__ (SVM)
# in the sense of a maximum-margin classifier. Other convex loss functions
# lead to more general variations of support vector machines.
#
# .. note::
#
#    Theory predicts that kernel-based training will always find better or equally good
#    models :math:`f_{\rm trained}` for the optimisation problem stated above. However, to show this here we would have
#    to either regularise the variational training by a term :math:`\mathrm{tr}\{\mathcal{M^2\}`, or switch off
#    regularisation in the classical SVM, which denies the SVM a lot of its strength.
#
#    The kernel-based and variational training in this demonstration therefore optimize slightly different cost
#    functions, and it is out of our scope to establish whether one training method finds better minimum than
#    the other.
#


######################################################################
# Kernel-based training
# =====================
#


######################################################################
# First, let’s import all sorts of useful methods:
#

import numpy as np
import torch
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor

import matplotlib.pyplot as plt

np.random.seed(42)


######################################################################
# The second step is to make an artificial toy data set.
#

X, y = make_blobs(n_samples=150, n_features=3, centers=2, cluster_std=0.7)

# scaling the inputs is important since the embedding we use is periodic
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# scaling the labels to -1, 1 is important for the SVM and the
# definition of a hinge loss
y_scaled = 2 * (y - 0.5)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)


######################################################################
# We will use the `amplitude embedding
# template <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.embeddings.AmplitudeEmbedding.html>`__
# which needs as many qubits as there are features.
#

n_qubits = len(X_train[0])


######################################################################
# To implement the kernel we could prepare the two states
# :math:`| \phi(x)\rangle`, :math:`| \phi(x')\rangle` on different qubits
# with amplitude embedding routines :math:`S(x), S(x')` and measure their
# overlap with a small routine called a SWAP test.
#
# However, we need only half the number of qubits if we prepare
# :math:`| \phi(x)\rangle` and then apply an inverse state preparation
# using :math:`x'` on the same qubits. We then measure the projector onto
# the initial state :math:`|0\rangle \langle 0|`.
#
# .. figure:: ../demonstrations/kernel_based_training/kernel_circuit.png 
#       :align: center
#       :scale: 100% 
#       :alt: Kernel evaluation circuit
#
# To verify that this gives us the kernel:
#
# .. math::  \langle 0 |S(x') S(x)^{\dagger} |0\rangle \langle 0| S(x')^{\dagger} S(x)  | 0\rangle  = | \langle \phi(x') | \phi(x)\rangle|^2 = \kappa(x, x').
#
# Note that a projector :math:`|0 \rangle \langle 0|` can be constructed
# as follows in PennyLane:
#
# .. code:: python
#
#    observables = [qml.PauliZ(i) for i in range(n_qubits)]
#    Tensor(*observables)
#
# Altogether, we use the following quantum node as a “quantum kernel
# evaluator”:
#

dev_kernel = qml.device("default.qubit", wires=n_qubits)

observables = [qml.PauliZ(i) for i in range(n_qubits)]


@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """
    The quantum kernel.
    """
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.inv(AngleEmbedding(x2, wires=range(n_qubits)))
    return qml.expval(Tensor(*observables))


######################################################################
# A good sanity check is whether measuring the distance between one and
# the same data point returns 1:
#

kernel(X_train[0], X_train[0])


######################################################################
# The way an SVM with a custom kernel is implemented in scikit-learn
# requires us to pass a function that computes a matrix of kernel
# evaluations for samples in two different datasets A, B.
#


def kernel_matrix(A, B):
    """
    Compute the matrix whose entries are the kernel
    evaluated on pairwise data from sets A and B.
    If A=B, this is the Gram matrix.
    """
    return np.array([[kernel(a, b) for b in B] for a in A])


######################################################################
# Training the SVM is a breeze in scikit-learn:
#

svm = SVC(kernel=kernel_matrix)
svm.fit(X_train, y_train)


######################################################################
# Let’s compute the accuracy on the test set.
#

predictions = svm.predict(X_test)
accuracy_score(predictions, y_test)


######################################################################
# How many times was the quantum device evaluated?
#

dev_kernel.num_executions


######################################################################
# This number can be derived as follows: For :math:`M` training samples,
# the SVM must construct the :math:`M \times M` dimensional kernel gram
# matrix for training. To classify :math:`M_{\rm pred}` new samples, the
# SVM needs to evaluate the kernel at most :math:`M_{\rm pred}M` times to get the
# pairwise distances between training vectors and test samples.
#
# Overall, the number of kernel evaluations of the above script should
# therefore roughly amount to:
#


def circuit_evals_kernel(n_data, split):
    """
    Compute how many circuit evaluations one needs for kernel-based training.
    """

    M = np.ceil(0.75 * n_data)
    Mpred = n_data - M

    n_training = M * M
    n_prediction = M * Mpred

    return n_training + n_prediction


circuit_evals_kernel(n_data=len(X), split=len(X_train) / len(X_test))


######################################################################
# A similar example using variational training
# ============================================
#


######################################################################
# Using the variational principle of training, we can propose an *ansatz*
# for the (circuit before the) measurement and train it directly. By
# increasing the number of layers of the ansatz, its expressivity
# increases. Depending on the ansatz, we can express any measurement, or
# only search through a subspace of all measurements for the best
# candidate.
#
# Remember from above, the variational training does not optimise
# *exactly* the same cost as the SVM, but we try to match them as closely
# as possible. For this we use a bias term in the quantum model, and train
# on the hinge loss.
#

dev_var = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev_var, interface="torch", diff_method="parameter-shift")
def quantum_model(x, params):
    """
    A variational circuit approximation of the quantum model.
    """

    # embedding
    AngleEmbedding(x, wires=range(n_qubits))

    # trainable measurement
    StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


def quantum_model_plus_bias(x, params, bias):
    """
    Adding a bias.
    """
    return quantum_model(x, params) + bias


def hinge_loss(predictions, targets):
    """
    Implements the hinge loss.
    """
    all_ones = torch.ones_like(targets)
    hinge_loss = all_ones - predictions * targets
    # trick: since the max function may not be diffable,
    # use the mathematically equivalent relu instead
    hinge_loss = relu(hinge_loss)
    return hinge_loss


######################################################################
# We now summarise the usual training and prediction steps into two
# functions that we can later call at will. Most of these functions
# convert between numpy and torch, which we need for the differentiable
# ``relu`` function used in the hinge loss.
#


def quantum_model_train(n_layers, steps, batch_size):
    """
    Train the quantum model defined above.
    """
    params = np.random.random((2, n_qubits, 3))
    params_torch = torch.tensor(params, requires_grad=True)
    bias_torch = torch.tensor(0.0)

    opt = torch.optim.Adam([params_torch, bias_torch], lr=0.1)

    loss_history = []
    for i in range(steps):

        batch_ids = np.random.choice(len(X_train), batch_size)

        X_batch = X_train[batch_ids]
        y_batch = y_train[batch_ids]

        X_batch_torch = torch.tensor(X_batch, requires_grad=False)
        y_batch_torch = torch.tensor(y_batch, requires_grad=False)

        def closure():
            opt.zero_grad()
            preds = torch.stack(
                [quantum_model_plus_bias(x, params_torch, bias_torch) for x in X_batch_torch]
            )
            loss = torch.mean(hinge_loss(preds, y_batch_torch))

            # bookkeeping
            current_loss = loss.detach().numpy().item()
            loss_history.append(current_loss)
            if i % 10 == 0:
                print("step", i, ", loss", current_loss)

            loss.backward()
            return loss

        opt.step(closure)

    return params_torch, bias_torch, loss_history


def quantum_model_predict(X_pred, trained_params, trained_bias):
    """
    Predict using the quantum model defined above.
    """
    p = []
    for x in X_pred:

        x_torch = torch.tensor(x)
        pred_torch = quantum_model_plus_bias(x_torch, trained_params, trained_bias)
        pred = pred_torch.detach().numpy().item()
        if pred > 0:
            pred = 1
        else:
            pred = -1

        p.append(pred)
    return p


######################################################################
# Let’s train the variational model and see how well we are doing on the
# test set.
#

n_layers = 1
batch_size = 20
steps = 80
trained_params, trained_bias, loss_history = quantum_model_train(n_layers, steps, batch_size)

pred_test = quantum_model_predict(X_test, trained_params, trained_bias)
print("accuracy on test set:", accuracy_score(pred_test, y_test))

plt.plot(loss_history)
plt.ylim((0, 1))
plt.show()


######################################################################
# How often was the device executed?
#

dev_var.num_executions


######################################################################
# Let’s do another calculation: In each optimisation step, the variational
# circuit needs to compute the partial derivative of all :math:`K`
# trainable parameters for each sample in a batch. Using parameter-shift
# rules (which is necessary for hardware), we require roughly 2 circuit
# evaluations per partial derivative. Prediction uses only one circuit
# evaluation per sample.
#
# We roughly get:
#


def circuit_evals_variational(n_data, n_params, evals_per_derivative, split, n_steps, batch_size):
    """
    Compute how many circuit evaluations are needed for variational training.
    """

    M = int(np.ceil(0.75 * n_data))
    Mpred = n_data - M

    n_training = n_params * steps * batch_size * evals_per_derivative
    n_prediction = Mpred

    return n_training + n_prediction


circuit_evals_variational(
    n_data=len(X),
    n_params=len(trained_params.flatten()),
    evals_per_derivative=2,
    split=len(X_train) / len(X_test),
    n_steps=steps,
    batch_size=batch_size,
)


######################################################################
# Which method sales better?
# ==========================
#


######################################################################
# In this small example, the kernel-based training trumps variational
# training in the number of circuit evaluations. But how does the overall scaling
# look like? Let us make the assumption that the number of steps and the
# number of parameters in variational circuit training grows linearly with
# the size of the data set, and choose sensible defaults for all other
# setting. This is what we get:
#

variational_training = []
kernelbased_training = []
x_axis = range(0, 10000, 100)
for M in x_axis:

    var = circuit_evals_variational(
        n_data=M, n_params=M, evals_per_derivative=2, split=0.75, n_steps=M, batch_size=20
    )
    variational_training.append(var)

    kernel = circuit_evals_kernel(n_data=M, split=0.75)
    kernelbased_training.append(kernel)


plt.plot(x_axis, variational_training, label="variational QML")
plt.plot(x_axis, kernelbased_training, label="kernel-based QML")
plt.xlabel("size of data set")
plt.ylabel("number of circuit evaluations")
plt.legend()
plt.show()


######################################################################
# Under the assumptions made, we can see that for data sets up to about
# :math:`4000` samples, kernel-based training uses *fewer* circuit
# evaluations to variational training. Only then the quadratic scaling of
# kernel methods takes over.
#
# As mentioned in `Schuld (2021) <https://arxiv.org/abs/2101.11020>`__, 
# early results from the quantum machine learning literature show that
# larger fault-tolerant quantum computers enable us in principle to reduce
# the quadratic scaling to linear scaling, which may make kernel methods a
# serious alternative to neural networks for big data processing one day.
#
