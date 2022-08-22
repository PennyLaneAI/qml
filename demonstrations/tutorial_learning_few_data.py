r"""
.. _learning_few_data:

Generalization in QML from few training data
============================================

.. meta::
    :property="og:description": Generalization of quantum machine learning models.
    :property="og:image": https://pennylane.ai/qml/_images/few_data_thumbnail.png

.. related::

    tutorial_local_cost_functions Alleviating barren plateaus with local cost functions

*Authors: Korbinian Kottmann, Luis Mantilla Calderon, Maurice Weber. Posted: 01 June 2022*

In this tutorial, we dive into the generalization capabilities of quantum machine learning models.
For the example of a Quantum Convolutional Neural Network (QCNN), we show how its generalization error behaves as a
function of the number of training samples. This demo is based on the paper
*"Generalization in quantum machine learning from few training data"*. by Caro et al. [#CaroGeneralization]_.

What is Generalization in (Q)ML?
------------------------
When optimizing a machine learning model, be it classical or quantum, we aim to maximize its performance over the data
distribution of interest, for example, images of cats and dogs. However, in practice, we are limited to a finite amount of
data, which is why it is necessary to reason about how our model performs on new, previously unseen data. The difference
between the model's performance on the true data distribution and the performance estimated from our training data is
called the *generalization error* and indicates how well the model has learned to generalize to unseen data.

.. figure:: /demonstrations/learning_few_data/true_vs_sample.png
    :width: 75%
    :align: center

It is good to know that generalization can be seen as a manifestation of the bias-variance trade-off: models that
perfectly fit the training data admit a low bias at the cost of a higher variance, and hence typically perform poorly on unseen
test data. In the classical machine learning community, this trade-off has been extensively
studied and has led to optimization techniques that favour generalization, for example, by regularizing models via
their variance [#NamkoongVariance]_.

Let us now dive deeper into generalization properties of quantum machine learning (QML) models. We start by describing
the typical data processing pipeline of a QML model. A classical data input :math:`x` is first encoded in a quantum
state via a mapping :math:`x \mapsto \rho(x)`. This encoded state is then processed through a quantum
channel :math:`\rho(x) \mapsto \mathcal{E}_\alpha(\rho(x))` with learnable parameters :math:`\alpha`. Finally, a measurement is performed on the resulting state
to get the final prediction. Now, the goal is to minimize the expected loss over the data generating distribution
:math:`P` indicating how well our model performs on new data. Mathematically, for a loss function :math:`\ell`, the
expected loss is given by

.. math:: R(\alpha) = \mathbb{E}_{(x,y)\sim P}[\ell(\alpha;\,x,\,y)]

where :math:`x` are the features and :math:`y` are the labels. In practice, as :math:`P` is generally 
unknown, this quantity has to be estimated from a finite amount of data. Given
a training set :math:`S = \{(x_i,\,y_i)\}_{i=1}^N`, we estimate the performance of our QML model by calculating the
average loss over the training set

.. math:: \hat{R}_S(\alpha) = \frac{1}{N}\sum_{i=1}^N \ell(\alpha;\,x_i,\,y_i)

which is referred to as the training loss and is an unbiased estimate of :math:`R(\alpha)`. This is only a proxy
to the true quantity of interest :math:`R(\alpha)` and their difference is called the generalization error

.. math:: \mathrm{gen}(\alpha) =  R(\alpha) - \hat{R}_S(\alpha)

which is the quantity that we explore in this tutorial. Keeping in mind the bias-variance trade-off, one would expect
that more complex models, i.e., models with a larger number of parameters, achieve a lower error on the training data
but a higher generalization error. Having more training data, on the other hand, leads to a better approximation of the
true expected loss and hence lower generalization error. This intuition is made precise in Ref. [#CaroGeneralization]_
where it is shown that :math:`\mathrm{gen}(\alpha)` roughly scales as :math:`\mathcal{O}(\sqrt{T / N})` where :math:`T`
is the number of parametrized gates and :math:`N` is the number of training samples.
"""

##############################################################################
# Generalization Bounds for Quantum Machine Learning Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# As hinted at earlier, we expect the generalization error to depend both on the richness of the model class, as well as
# on the amount of training data available. As a first result, the authors of Ref. [#CaroGeneralization]_ found that for
# a QML model with at most :math:`T` parametrized local quantum channels, the generalization error depends on :math:`T`
# and :math:`N` according to
#
# .. math:: \mathrm{gen}(\alpha) \in \mathcal{O}\left(\sqrt{\frac{T\log T}{N}}\right).
#
# We see that this scaling is in line with our intuition that the generalization error scales inversely with the number
# of training samples and increases with the number of parametrized gates. However, as is the case for 
# quantum convolutional neural networks, it is possible to get a more fine-grained bound by including knowledge on the number :math:`M` of gates which have been reused (i.e. whose parameters are shared across wires). Naively, one could suspect that the generalization error scales as
# :math:`\tilde{\mathcal{O}}(\sqrt{MT/N})` by directly applying the above result (and where
# :math:`\tilde{\mathcal{O}}` includes logarithmic factors). However, the authors of Ref. [#CaroGeneralization]_ found
# that such models actually adhere to the better scaling
#
# .. math:: \mathrm{gen}(\alpha) \in \mathcal{O}\left(\sqrt{\frac{T\log MT}{N}}\right).
#
# With this, we see that for QCNNs to have a generalization error :math:`\mathrm{gen}(\alpha)\leq\epsilon`, we need a
# training set of size :math:`N \sim T \log MT / \epsilon^2`. For the special case of QCNNs, we can explicitly connect
# the number of samples needed for good generalization to the system size :math:`n` since these models
# use :math:`\mathcal{O}(\log(n))` independently parametrized gates, each of which is used at most :math:`n` times [#CongQuantumCNN]_.
# Putting the pieces together, we find that a training set of size
#
# .. math::  N \in \mathcal{O}(\mathrm{poly}(\log n))
#
# is sufficient for the generalization error to be bounded by :math:`\mathrm{gen}(\alpha) \leq \epsilon`.
# In the next part of this tutorial, we will illustrate this result by implementing a QCNN to classify different
# digits in the classical ``digits`` dataset. Before that, we set up our QCNN.

##############################################################################
# Quantum convolutional neural network
# ------------------------------------
# Before we start building a quantum CNN, let us remember the idea of their classical counterpart.
# Classical CNNs are a family of neural networks which make use of convolutions and pooling operations to
# insert an inductive bias, favoring invariances to spatial transformation like translations, rotations and scaling.
# In particular, one uses what is known as a *convolutional layer*,
# which consists of a small kernel (a window) that sweeps a 2D array (an image) and extracts local
# information about such an array. In addition, depending on the purpose of your CNN, one might want
# to make classification or feature predictions, which are arrays much smaller than the original image.
# To deal with this dimensionality difference, one uses what is known as a *pooling layer*. These
# layers are used to reduce the dimensionality of the 2D array being processed (whereas inverse pooling increases the
# dimensionality of a 2D array). Finally, one takes these two layers and applies them repeatedly and
# interchangeably as show in the figure below. 
#
# .. figure:: /demonstrations/learning_few_data/cnn_pic.png
#     :width: 75%
#     :align: center
#
# We want to build something similar for a quantum circuit. First, we import the necessary 
# libraries we will need in this demo and set a seed for reproducibility.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from tqdm.auto import trange

import jax
import jax.numpy as jnp

import optax # optimization using jax

import pennylane as qml
import pennylane.numpy as pnp

sns.set()

seed = 0
rng = np.random.default_rng(seed=seed)

##############################################################################
# To construct a convolutional and pooling layer in a quantum circuit, we will
# follow the QCNN construction proposed by [#CongQuantumCNN]_. The former layer
# will extract local correlations, while the latter allows reducing the dimensionality
# of the feature vector. In a qubit circuit, the convolutional layer, consisting of a kernel swept
# along the entire image, is now translated to a two-qubit unitary that correlates neighboring
# qubits.  As for the pooling layer, we will use a conditioned single-qubit unitary that depends
# on the measurement of a neighboring qubit. Finally, we use a *dense layer* that entangles all
# qubits of the final state using an all-to-all unitary gate.
#
# Breaking down the layers
# --------------------------
#
# The convolutional layer should have as an input the weights of the two-qubit unitary, which are
# to be updated in each training round.  In PennyLane, we model this arbitrary two-qubit unitary
# with a particular sequence of gates: two single-qubit gates :class:`~.pennylane.U3` (parametrized by three
# parameters, each), followed by three Ising interactions between both qubits (each interaction is
# parametrized by one parameter), and end with two additional :class:`~.pennylane.U3` gates on each of the two
# qubits.

def convolutional_layer(weights, wires, skip_first_layer=True):
    """Adds a convolutional layer to a circuit.
    Args:
        weights (np.array): 1D array with 15 weights of the parametrized gates.
        wires (list[int]): Wires where the convolutional layer acts on.
        skip_first_layer (bool): Skips the first two U3 gates of a layer."""
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

##############################################################################
# The pooling layer has as inputs the weights of the single-qubit conditional unitaries, which in
# this case, are :class:`~.pennylane.U3` gates. Then, we apply these conditional measurements to half of the
# unmeasured wires, reducing our system size by half.

def pooling_layer(weights, wires):
    """Adds a pooling layer to a circuit.
    Args:
        weights (np.array): Array with the weights of the conditional U3 gate.
        wires (list[int]): List of wires to apply the pooling layer on.
    """
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"

    for indx, w in enumerate(wires):
        if indx % 2 == 1 and indx < n_wires:
            m_outcome = qml.measure(w)
            qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])


##############################################################################
# We can construct a quantum CNN by combining both layers and using an arbitrary unitary to model
# a dense layer. It will take as input a set of features (the image), encode these features using
# an embedding map, apply rounds of convolutional and pooling layers, and eventually get the
# desired measurement statistics of the circuit.

def conv_and_pooling(kernel_weights, n_wires):
    """Apply both the convolutional and pooling layer."""
    convolutional_layer(kernel_weights[:15], n_wires)
    pooling_layer(kernel_weights[15:], n_wires)


def dense_layer(weights, wires):
    """Apply an arbitrary unitary gate to a specified set of wires."""
    qml.ArbitraryUnitary(weights, wires)

num_wires = 6
device = qml.device('default.qubit', wires=num_wires)

@qml.qnode(device, interface="jax")
def conv_net(weights, last_layer_weights, features):
    """Define the QCNN circuit
    Args:
        weights (np.array): Parameters of the convolution and pool layers.
        last_layer_weights (np.array): Parameters of the last dense layer.
        features (np.array): Input data to be embedded using AmplitudEmbedding."""

    layers = weights.shape[1]
    wires = list(range(num_wires))

    # inputs the state input_state
    qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5)

    # adds convolutional and pooling layers
    for j in range(layers):
        conv_and_pooling(weights[:, j], wires)
        wires = wires[::2]

    assert (
            last_layer_weights.size == 4 ** (len(wires)) - 1
    ), f"The size of the last layer weights vector is incorrect! \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
    dense_layer(last_layer_weights, wires)
    return qml.probs(wires=(0))


##############################################################################
# In the problem we will address, we need to encode 64 features
# in the state to be processed by the QCNN. Thus, we require six qubits to encode
# each feature value in the amplitude of each computational basis state.
#
# Training the QCNN on the digits dataset
# ---------------------------------------
# In this demo, we are going to classify the digits ``0`` and ``1`` from the classical ``digits`` dataset.
# These are 8 by 8 pixel arrays of hand-written digits as shown below.

digits = datasets.load_digits()
images, labels = digits.data, digits.target

images = images[np.where((labels == 0) | (labels == 1))]
labels = labels[np.where((labels == 0) | (labels == 1))]

fig, axes = plt.subplots(nrows=1,ncols=12, figsize=(3, 3));

for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i].reshape((8,8)), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

##############################################################################
# For convenience, we create a ``load_digits_data`` function that will make random training and
# testing sets from the ``digits`` dataset from ``sklearn.dataset``.

def load_digits_data(num_train, num_test, rng):
    """Return training and testing data of digits dataset."""
    digits = datasets.load_digits()
    features, labels = digits.data, digits.target

    # only use first two classes
    features = features[np.where((labels == 0) | (labels == 1))]
    labels = labels[np.where((labels == 0) | (labels == 1))]

    # normalize data
    features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))

    # subsample train and test split
    train_indices = rng.choice(len(labels), num_train, replace=False)
    test_indices = rng.choice(np.setdiff1d(range(len(labels)), train_indices), num_test, replace=False)

    x_train, y_train = features[train_indices], labels[train_indices]
    x_test, y_test = features[test_indices], labels[test_indices]

    return jnp.asarray(x_train), jnp.asarray(y_train), jnp.asarray(x_test), jnp.asarray(y_test)

##############################################################################
# To optimize the weights of our variational model, we define the cost and accuracy functions
# to train and quantify the performance on the classification task of the previously described QCNN.

@jax.jit
def compute_out(weights, weights_last, features, labels):
    """Computes the output of the corresponding label in the qcnn"""
    cost = lambda weights, weights_last, feature, label: conv_net(weights, weights_last, feature)[label]
    return jax.vmap(cost, in_axes=(None, None, 0, 0), out_axes=0)(weights, weights_last, features, labels)

def compute_accuracy(weights, weights_last, features, labels):
    """Computes the accuracy over the provided features and labels"""
    out = compute_out(weights, weights_last, features, labels)
    return jnp.sum(out > 0.5)/len(out)

def compute_cost(weights, weights_last, features, labels):
    """Computes the cost over the provided features and labels"""
    out = compute_out(weights, weights_last, features, labels)
    return 1.0 - jnp.sum(out) / len(labels)

def init_weights():
    """Initializes random weights for the QCNN model."""
    weights = pnp.random.normal(loc=0, scale=1, size=(18, 2), requires_grad=True)
    weights_last = pnp.random.normal(loc=0, scale=1, size=4 ** 2 - 1, requires_grad=True)
    return jnp.array(weights), jnp.array(weights_last)

value_and_grad = jax.jit(jax.value_and_grad(compute_cost, argnums=[0, 1]))

##############################################################################
# We are going to perform the classification for differently sized training sets. Therefore, we
# define the classification procedure once and then perform it for different datasets.
# Finally, we update the weights using the :class:`pennylane.AdamOptimizer` and use these updated weights to
# calculate the cost and accurracy on the testing and training set.

def train_qcnn(n_train, n_test, n_epochs):
    """
    Args:
        n_train  (int): number of training examples
        n_test   (int): number of test examples
        n_epochs (int): number of training epochs
        desc  (string): displayed string during optimization

    Returns:
        dict: n_train, steps, train_cost_epochs, train_acc_epochs, test_cost_epochs, test_acc_epochs

    """
    # load data
    x_train, y_train, x_test, y_test = load_digits_data(n_train, n_test, rng)

    # init weights and optimizer
    weights, weights_last = init_weights()

    # learning rate decay
    cosine_decay_scheduler = optax.cosine_decay_schedule(0.1, decay_steps=n_epochs, alpha=0.95)
    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
    opt_state = optimizer.init((weights, weights_last))

    # data containers
    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

    for step in range(n_epochs):
        # Training step with (adam) optimizer
        train_cost, grad_circuit = value_and_grad(weights, weights_last, x_train, y_train)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        weights, weights_last = optax.apply_updates((weights, weights_last), updates)

        train_cost_epochs.append(train_cost)

        # compute accuracy on training data
        train_acc = compute_accuracy(weights, weights_last, x_train, y_train)
        train_acc_epochs.append(train_acc)

        # compute accuracy and cost on testing data
        test_out = compute_out(weights, weights_last, x_test, y_test)
        test_acc = jnp.sum(test_out > 0.5)/len(test_out)
        test_acc_epochs.append(test_acc)
        test_cost = 1.0 - jnp.sum(test_out) / len(test_out)
        test_cost_epochs.append(test_cost)

    return dict(
        n_train=[n_train] * n_epochs,
        step=np.arange(1, n_epochs+1, dtype=int),
        train_cost=train_cost_epochs,
        train_acc=train_acc_epochs,
        test_cost=test_cost_epochs,
        test_acc=test_acc_epochs
    )

##############################################################################
# .. note::
#
#     There are some small intricacies for speeding up this code that are worth mentioning: We are using ``jax`` for our training
#     because it allows for just-in-time (``jit``) compilation, see `jax docs <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_. A function decorated with ``@jax.jit`` will be compiled upon its first execution
#     and cached for future executions. This means the first execution will take longer, but all subsequent executions are substantially faster.
#     Further, we use ``jax.vmap`` to vectorize the execution of the QCNN over all input states (as opposed to looping through the training and test set at every execution)

##############################################################################
# Training for different training set sizes yields different accuracies, as seen below. As we increase the training data size, the overall test accuracy,
# a proxy for the models' generalization capabilities, increases.

n_test = 100
n_epochs = 100
n_reps = 100

def run_iterations(n_train):
    results_df = pd.DataFrame(columns=['train_acc', 'train_cost', 'test_acc', 'test_cost', 'step', 'n_train'])
    pbar = trange(n_reps, desc='train qcnn')

    for _ in pbar:
        results = train_qcnn(n_train=n_train, n_test=n_test, n_epochs=n_epochs)
        results_df = pd.concat([results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True)

    return results_df

results_df = run_iterations(n_train=40)

##############################################################################
# Finally, we plot the loss and accuracy for both the training and testing set
# for all training epochs, and compare the test and train accuracy of the model.

def make_plot(df, n_train):
    fig, axs = plt.subplots(ncols=3, figsize=(14,5))

    df_agg = df.groupby(['step']).agg(['mean', 'std'])

    # plot epoch vs loss
    ax = axs[0]
    ax.plot(df_agg.index, df_agg.train_cost['mean'], "o--", label="train", markevery=10)
    ax.plot(df_agg.index, df_agg.test_cost['mean'], "x--", label="test", markevery=10)
    ax.set_ylabel("loss", fontsize=18)
    ax.set_xlabel("epoch", fontsize=18)
    ax.legend(fontsize=14)

    # plot epoch vs acc (train + test)
    ax = axs[1]
    ax.plot(df_agg.index, df_agg.train_acc['mean'],"o:", label=f"train", markevery=10)
    ax.plot(df_agg.index, df_agg.test_acc['mean'],"x--", label=f"test", markevery=10)
    ax.set_ylabel("accuracy", fontsize=18)
    ax.set_xlabel("epoch", fontsize=18)
    ax.legend(fontsize=14)


    # plot train acc vs test acc
    ax = axs[2]
    ax.scatter(df.train_acc, df.test_acc, alpha=0.1, marker='D')
    beta, m = np.polyfit(np.array(df.train_acc, dtype=float), np.array(df.test_acc, dtype=float), 1)
    reg = np.poly1d([beta, m])
    ax.plot(df.train_acc, reg(np.array(df.train_acc, dtype=float)),"-", color='black', lw=0.75)
    ax.set_ylabel("test accuracy", fontsize=18)
    ax.set_xlabel("train accuracy", fontsize=18)

    fig.suptitle(f'Performance Measures for Training Set of Size $N=${n_train}', fontsize=20)
    plt.tight_layout()
    plt.show()

make_plot(results_df, n_train=40)

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
##############################################################################
# .. bio:: Korbinian Kottmann
#    :photo: ../_static/authors/qottmann.jpg
#
#    Korbinian is a summer resident at Xanadu, interested in (quantum) software development, quantum computing and (quantum) machine learning.

##############################################################################
# .. bio:: Luis Mantilla Calderon
#    :photo: ../_static/authors/qottmann.jpg
#
#    Luis is a cool dude studying in Vancouver, he likes to lose money on crypto and the stock market

##############################################################################
# .. bio:: Maurice Weber
#    :photo: ../_static/authors/qottmann.jpg
#
#    Maurice is a cool dude from ETH that does fancy computer science stuff!