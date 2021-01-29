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
    
*Author: PennyLane dev team. Posted: XX Feb 2021. Last updated: XX Feb 2021.*

This demonstration illustrates how one can train quantum machine
learning models with a kernel-based approach instead of the usual
`variational
approach <https://pennylane.ai/qml/glossary/variational_circuit.html>`__.
The theoretical background has been established in many papers in the literature 
such as `Havlicek et al. (2018) <https://arxiv.org/abs/1804.11326>`__, `Schuld and Killoran (2018) <https://arxiv.org/abs/1803.07128>`__,
`Liu et al. (2020) <https://arxiv.org/abs/2010.02174>`__, `Huang et al. (2020) <https://arxiv.org/pdf/2011.01938.pdf>`__,
and has been systematically summarised in the overview `Schuld (2021) <https://arxiv.org/abs/2101.11020>`__ which we follow here.

As an example of kernel-based training we use a combination of PennyLane
and the powerful `scikit-learn <https://scikit-learn.org/>`__ machine
learning library to use a support vector machine with 
a *quantum kernel*. We then compare this strategy with a variational
quantum circuit trained via stochastic gradient descent using
PyTorch.

A secondary goal of the demo is to estimate the number of circuit evaluations needed in
both approaches. For the example used here, kernel-based training requires fewer (and shorter) 
quantum circuit evaluations. 

More generally, we will see that the comparison with variational training depends how the number of variational parameters 
scales with the amount of data. If variational circuits turn out to be similar to neural networks and grow linearly in size 
with the data, kernel-based training is much more efficient. 
If instead the number of parameters plateaus with growing data sizes, variational training would require fewer circuit 
evaluations, although it will never be competitive with classical neural networks. 

.. figure::  ../demonstrations/kernel_based_training/scaling.png 
       :align: center
       :scale: 100%
       :alt: Scaling of kernel-based vs. variational learning

After working through this demo, the reader should:

* be able to use a support vector machine with a quantum kernel computed with PennyLane, and

* understand the scaling of quantum circuit evaluations required in kernel-based versus 
  variational training.  


"""

######################################################################
# Background
# ==========
#
# Let us consider a *quantum model* of the form
#
# .. math:: f(x) = \langle \phi(x) | \mathcal{M} | \phi(x)\rangle,
#
# where :math:`| \phi(x)\rangle` is prepared
# by a fixed `embedding
# circuit <https://pennylane.readthedocs.io/en/stable/introduction/templates.html#intro-ref-temp-emb>`__ that 
# encodes data inputs :math:`x`,
# and :math:`\mathcal{M}` is an arbitrary observable. This model includes variational 
# quantum machine learning models, since the observable can
# effectively be implemented by a simple measurement that is preceded by a
# variational circuit: 
#
#
# .. figure:: ../demonstrations/kernel_based_training/quantum_model.png 
#       :align: center
#       :scale: 30%
#       :alt: quantum-model
#
#
# For example, applying a circuit :math:`G(\theta)` and then
# measuring the Pauli-Z observable :math:`\sigma^0_z` of the first qubit
# implements the trainable measurement 
# :math:`\mathcal{M}(\theta) = G^{\dagger}(\theta) \sigma^0_z G(\theta)`.
#
# The main practical consequence of approaching quantum machine learning with a 
# kernel approach is that instead of training :math:`f` variationally,
# we can often train an equivalent classical kernel method with a kernel executed on a
# quantum device. This *quantum kernel*
# is given by the mutual overlap of two data-encoding quantum states,
#
# .. math::  \kappa(x, x') = | \langle \phi(x') | \phi(x)\rangle|^2.
#
# Kernel-based training therefore by-passes the variational part and
# measurement of common variational circuits, and only depends on the
# embedding.
#
# If the loss function in training is the `hinge
# loss <https://en.wikipedia.org/wiki/Hinge_loss>`__, the kernel method
# corresponds to a standard `support vector
# machine <https://en.wikipedia.org/wiki/Support-vector_machine>`__ (SVM)
# in the sense of a maximum-margin classifier. Other convex loss functions
# lead to more general variations of support vector machines.
#
# .. note::
#
#    More precisely, we can replace variational with kernel-based 
#    training if the optimisation
#    problem can be written as minimizing a cost of the form
# 
#    .. math::  \min_f  \lambda \mathrm{tr}\{\mathcal{M}^2\} + \frac{1}{M}\sum_{m=1}^M L(f(x^m), y^m), 
#
#    which is a regularized empirical risk with training data samples :math:`(x^m, y^m)_{m=1\dots M}`, 
#    regularisation strength :math:`\lambda \in \mathbb{R}`, and loss function :math:`L`.
#
#    Theory predicts that kernel-based training will always find better or equally good
#    minima of this risk. However, to show this here we would have
#    to either regularise the variational training by the trace of the squared observable, or switch off
#    regularisation in the classical SVM, which denies it a lot of its strength. The kernel-based and the variational 
#    training in this demonstration therefore optimize slightly different cost
#    functions, and it is out of our scope to establish whether one training method finds a better minimum than
#    the other.
#


######################################################################
# Kernel-based training
# =====================
#
# First, we will turn to kernel-based training of quantum models. 
# As stated above, an example implementation is a standard support vector 
# machine with a kernel computed by a quantum circuit.
# 


######################################################################
# First, let’s import all sorts of useful methods:
#

import numpy as np
import torch
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor

import matplotlib.pyplot as plt

np.random.seed(42)


######################################################################
# The second step is to make a toy data set.
#

X, y = load_iris(return_X_y=True)

# turn into 2 classes
X = X[:100]
y = y[:100]

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
n_qubits


######################################################################
# To implement the kernel we could prepare the two states
# :math:`| \phi(x)\rangle`, :math:`| \phi(x')\rangle` on different sets of qubits
# with amplitude embedding routines :math:`S(x), S(x')` and measure their
# overlap with a small routine called a `SWAP test <https://en.wikipedia.org/wiki/Swap_test>`__.
#
# However, we need only half the number of qubits if we prepare
# :math:`| \phi(x)\rangle` and then apply the inverse embedding
# with :math:`x'` on the same qubits. We then measure the projector onto
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
#    projector = Tensor(*observables)
#
# Altogether, we use the following quantum node as a *quantum kernel
# evaluator*:
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
# A good sanity check is whether evaluating the kernel of a data point and 
# itself returns 1:
#

kernel(X_train[0], X_train[0])


######################################################################
# The way an SVM with a custom kernel is implemented in scikit-learn
# requires us to pass a function that computes a matrix of kernel
# evaluations for samples in two different datasets A, B. If A=B, 
# this is the `Gram matrix <https://en.wikipedia.org/wiki/Gramian_matrix>`__.
#


def kernel_matrix(A, B):
    """
    Compute the matrix whose entries are the kernel
    evaluated on pairwise data from sets A and B.
    """
    return np.array([[kernel(a, b) for b in B] for a in A])


######################################################################
# Training the SVM is a breeze in scikit-learn:
#

svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)


######################################################################
# Let’s compute the accuracy on the test set.
#

predictions = svm.predict(X_test)
accuracy_score(predictions, y_test)


######################################################################
# The SVM predicted all test points correctly.
#
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
# .. note:: 
#    
#     Depending on the implementation of the SVM, only :math:`S \leq M_{\rm pred}`
#     *support vectors* are needed. 
#
# Overall, the number of kernel evaluations of the above script should
# therefore roughly amount to at most:
#


def circuit_evals_kernel(n_data, split):
    """
    Compute how many circuit evaluations one needs for kernel-based training and prediction.
    """

    M = int(np.ceil(split * n_data))
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
# increases. Depending on the ansatz, we may only
# search through a subspace of all measurements for the best
# candidate.
#
# Remember from above, the variational training does not optimize
# *exactly* the same cost as the SVM, but we try to match them as closely
# as possible. For this we use a bias term in the quantum model, and train
# on the hinge loss.
#

dev_var = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev_var, interface="torch", diff_method="parameter-shift")
def quantum_model(x, params):
    """
    A variational quantum model.
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
    # trick: since the max function is not diffable,
    # use the mathematically equivalent relu instead
    hinge_loss = relu(hinge_loss)
    return hinge_loss


######################################################################
# We now summarise the usual training and prediction steps into two
# functions that we can later call at will. Most of the work is to
# convert between numpy and torch, which we need for the differentiable
# ``relu`` function used in the hinge loss.
#


def quantum_model_train(n_layers, steps, batch_size):
    """
    Train the quantum model defined above.
    """
    params = np.random.random((n_layers, n_qubits, 3))
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

n_layers = 2
batch_size = 20
steps = 100
trained_params, trained_bias, loss_history = quantum_model_train(n_layers, steps, batch_size)

pred_test = quantum_model_predict(X_test, trained_params, trained_bias)
print("accuracy on test set:", accuracy_score(pred_test, y_test))

plt.plot(loss_history)
plt.ylim((0, 1))
plt.show()


######################################################################
# The variational circuit has a slightly lower 
# accuracy than the SVM - but this depends very much on the training settings 
# we used. Different random parameter initialisations may indeed get 
# perfect test accuracy.
#
# How often was the device executed?
#

dev_var.num_executions


######################################################################
# That is a lot more than the kernel method took!
#
# Let’s try to understand this value. In each optimization step, the variational
# circuit needs to compute the partial derivative of all
# trainable parameters for each sample in a batch. Using `parameter-shift
# rules <https://pennylane.ai/qml/glossary/parameter_shift.html>`__ we require roughly 2 circuit
# evaluations per partial derivative. Prediction uses only one circuit
# evaluation per sample.
#
# This would result in:
#


def circuit_evals_variational(n_data, n_params, n_steps, shift_terms, split, batch_size):
    """
    Compute how many circuit evaluations are needed for variational training and prediction.
    """

    M = int(np.ceil(split * n_data))
    Mpred = n_data - M

    n_training = n_params * n_steps * batch_size * shift_terms
    n_prediction = Mpred

    return n_training + n_prediction


circuit_evals_variational(
    n_data=len(X),
    n_params=len(trained_params.flatten()),
    n_steps=steps,
    shift_terms=2,
    split=len(X_train) / len(X_test),
    batch_size=batch_size,
)


######################################################################
# The estimate is a bit higher because it does not account for some optimizations
# that PennyLane performs under the hood.
#
# It is important to note that while they are trained in a similar manner, 
# the number of variational circuit evaluations differs from the number of 
# neural network model evaluations in classical machine learning, which would be given by:
#

def model_evals_nn(n_data, n_params, n_steps, split, batch_size):
    """
    Compute how many model evaluations are needed for neural network training and prediction.
    """

    M = int(np.ceil(split * n_data))
    Mpred = n_data - M

    n_training = n_steps * batch_size
    n_prediction = Mpred

    return n_training + n_prediction
    
######################################################################
# In each step of neural network training, due to the power of automatic differentiation, 
# the backpropagation algorithm can compute a 
# gradient for all parameters in (more-or-less) a single run. 
# For all we know at this stage, the no-cloning principle prevents variational circuits from using these tricks, 
# which leads to ``n_training`` in ``circuit_evals_variational`` depending on the number of parameters, but not in 
# ``model_evals_nn``. 
#
# For the same example as used here, a neural network would therefore 
# have far fewer model evaluations than both variational and kernel-based training:
#

model_evals_nn(
    n_data=len(X),
    n_params=len(trained_params.flatten()),
    n_steps=steps,
    split=len(X_train) / len(X_test),
    batch_size=batch_size,
)
   

######################################################################
# Which method scales best?
# =========================
#


######################################################################
# Of course, the answer to this question depends on how the variational model 
# is set up, and we need to make a few assumptions: 
#
# 1. Even if we use single-batch stochastic gradient descent, in which every training step uses 
#    exactly one training sample, we would want to see every training sample at least once on average. 
#    Therefore, the number of steps should scale linearly with the number of training data. 
#
# 2. Modern neural networks often have many more parameters than training
#    samples. But we do not know yet whether variational circuits really need that many parameters as well.
#    We will therefore use two cases for comparison: 
#
#    2a) the number of parameters grows linearly with the training data, or ``n_params = M``, 
#
#    2b) the number of parameters grows as the square root with the training data, or ``n_params = np.sqrt(M)``. 
#
# Note that compared to the example above with 75 training samples and 24 parameters, a) overestimates the number of evaluations, while b) 
# underestimates it.
#


######################################################################
# This is how the three methods compare:
#

variational_training1 = []
variational_training2 = []
kernelbased_training = []
nn_training = []
x_axis = range(0, 2000, 100)

for M in x_axis:

    var1 = circuit_evals_variational(
        n_data=M, n_params=M, n_steps=M,  shift_terms=2, split=0.75, batch_size=1
    )
    variational_training1.append(var1)

    var2 = circuit_evals_variational(
        n_data=M, n_params=round(np.sqrt(M)), n_steps=M,  shift_terms=2, split=0.75, batch_size=1
    )
    variational_training2.append(var2)
    
    kernel = circuit_evals_kernel(n_data=M, split=0.75)
    kernelbased_training.append(kernel)
    
    nn = model_evals_nn(
        n_data=M, n_params=M, n_steps=M, split=0.75, batch_size=1
    )
    nn_training.append(nn)


plt.plot(x_axis, nn_training, linestyle='--', label="neural net")
plt.plot(x_axis, variational_training1, label="var. circuit (linear param scaling)")
plt.plot(x_axis, variational_training2, label="var. circuit (srqt param scaling)")
plt.plot(x_axis, kernelbased_training, label="(quantum) kernel")
plt.xlabel("size of data set")
plt.ylabel("number of evaluations")
plt.legend()
plt.tight_layout()
plt.show()



######################################################################
# This is the plot we saw at the beginning. Under the assumptions made, 
# one can see that the relation between kernel-based 
# and variational quantum machine learning depends on how many parameters the latter needs: 
# If variational circuits turn out to be as parameter-hungry as neural networks,
# kernel-based training will consistently outperform it. 
# However, if we find ways to train variational circuits with fewer parameters,
# they can use this as a means to catch up with the good scaling neural networks draw from backpropagation.   
#
# What we learn from this demo is that unless your variational circuit has many fewer 
# parameters than training data, kernel methods could be a much faster alternative!
#
# Finally, it is important to note that fault-tolerant quantum computers may change the picture significantly. 
# As mentioned in `Schuld (2021) <https://arxiv.org/abs/2101.11020>`__, 
# early results from the quantum machine learning literature show that
# larger quantum computers enable us in principle to reduce
# the quadratic scaling of kernel methods to linear scaling, which may make kernel methods a
# serious alternative to even neural networks for big data processing one day.
#
