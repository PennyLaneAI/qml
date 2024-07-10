r"""Post-Variational Quantum Neural Networks
========================================
"""

######################################################################
# Variational algorithms are proposed to solve optimization problems in chemistry, combinatorial
# optimization and machine learning, with potential quantum advantage. [#cerezo2021variational]_ Such algorithms operate by
# first encoding data :math:`x` into a :math:`n`-qubit quantum state. The quantum state is then
# transformed by an Ansatz :math:`U(\theta)`. The parameters :math:`\theta` are optimized by
# evaluating gradients of the quantum circuit via parameter-shift rules and calculating updates of the
# parameter via optimization on classical computers.
#
# However, many Ansatze face the barren plateau problem [#mcclean2018barren]_, which leads to difficulty in convergence
# using gradient-based optimization techniques. With the general difficulty and lack of training
# gurantees provided by variational algorithms, we discuss alternative strategies derived from the
# variational method as the theoretical basis for optimisation but avoid tuning the parameterised
# quantum states.
#
# In this demo, we discuss “post-variational strategies”, proposed in this
# `paper <https://arxiv.org/pdf/2307.10560>`__, where we take the classical combination of multiple
# fixed quantum circuits and find the optimal combination through feeding our combinations through a
# classical multilayer perceptron. We shift tunable parameters from the quantum computer to the
# classical computer, opting for ensemble strategies when optimizing quantum models. This exchanges
# expressibility of the circuit with trainability of the entire model. Below, we discuss various
# strategies and design principles for constructing individual quantum circuits, where the resulting
# ensembles can be optimized with classical optimisation methods. [#huang2024postvariational]_
#

######################################################################
# .. figure:: ../_static/demonstration_assets/PVQNN/PVdrawing.jpeg
#    :align: center
#    :width: 90%

######################################################################
# We compare our post-variational strategies to the conventional variational neural network in the
# table below.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/PVQNN/table.png
#    :align: center
#    :width: 90%

######################################################################
# This example demonstrates how to employ our post variational quantum neural network on the classical
# machine learning task of image classification. Here, we solve the problem of identifying handwritten
# digits of threes and fives and obtain training performance better than that of variational
# algorithms.
#

import pennylane as qml
from pennylane.templates import BasicEntanglerLayers
from pennylane import numpy as np

######################################################################
# Data Preprocessing
# ------------------
#

from tqdm import tqdm
import jax
from jax import numpy as jnp
import optax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np

######################################################################
# We train our models on the digits dataset, which we import using sklearn. The dataset has grescale
# images of size :math:`8\times 8` pixels. We only consider the digits ‘3’ and ‘5’, and standardise
# the labels. There are 273 images for training and 91 images for testing. Each feature is transformed
# into a 8 by 8 grid, and each target is standardised.
#

X_digits, y_digits = load_digits(n_class=6, return_X_y=True)
filter_mask = np.isin(y_digits, [3, 5])
X_digits = X_digits[filter_mask]
y_digits = y_digits[filter_mask]
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.25, random_state=16
)
X_train = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_train])
X_test = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_test])
y_train = y_train - 4
y_test = y_test - 4

######################################################################
# A visualization of one data point is shown below.
#

plt.gray()
plt.matshow(X_train[6])
print(y_train[6])
plt.show()

######################################################################
# Variational Algorithm
# ---------------------
#

######################################################################
# As a baseline comparison, we first test the performance of a shallow variational algorithm on the
# digits dataset shown above.
#

######################################################################
# For the feature map, each column of the image is encoded into a single qubit, and each row is
# encoded consecutively via alternating rotation-Z and rotation-X gates.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/PVQNN/featuremap.png
#    :align: center
#    :width: 90%

######################################################################
# This Ansatz is also used as the Ansatze generating backbone for the Ansatz expansion and hybrid
# post-variational strategies. When we set all initial parameters to 0, the Ansatz evaluates to
# identity.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/PVQNN/ansatz.png
#    :align: center
#    :width: 90%

######################################################################
# We write code for the above ansatz and feature map as shown below.
#


def feature_map(features):
    for i in range(len(features[0])):
        qml.Hadamard(i)
    for i in range(len(features)):
        if i % 2:
            qml.AngleEmbedding(features=features[i], wires=range(8), rotation="Z")
        else:
            qml.AngleEmbedding(features=features[i], wires=range(8), rotation="X")


def ansatz(params):
    for i in range(8):
        qml.RY(params[i], wires=i)
    for i in range(8):
        qml.CNOT(wires=[(i - 1) % 8, (i) % 8])
    for i in range(8):
        qml.RY(params[i + 8], wires=i)
    for i in range(8):
        qml.CNOT(wires=[(8 - 2 - i) % 8, (8 - i - 1) % 8])


######################################################################
# We then build the quantum node by combining the above feature map and ansatz.
#

dev = qml.device("default.qubit", wires=8)


@qml.qnode(dev)
def circuit(params, features):
    feature_map(features)
    ansatz(params)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def accuracy(labels, predictions):
    acc = sum([np.sign(l) == np.sign(p) for l, p in zip(labels, predictions)])
    acc = acc / len(labels)
    return acc


def cost(params, X, Y):
    predictions = [variational_classifier(params["weights"], params["bias"], x) for x in X]
    return square_loss(Y, predictions)


def acc(params, X, Y):
    predictions = [variational_classifier(params["weights"], params["bias"], x) for x in X]
    return accuracy(Y, predictions)


np.random.seed(0)
weights = 0.01 * np.random.randn(16)
bias = jnp.array(0.0)
params = {"weights": weights, "bias": bias}
opt = optax.adam(0.05)
batch_size = 7
num_batch = X_train.shape[0] // batch_size
opt_state = opt.init(params)
X_batched = X_train.reshape([-1, batch_size, 8, 8])
y_batched = y_train.reshape([-1, batch_size])


@jax.jit
def update_step_jit(i, args):
    params, opt_state, data, targets, batch_no = args
    _data = data[batch_no % num_batch]
    _targets = targets[batch_no % num_batch]
    loss_val, grads = jax.value_and_grad(cost)(params, _data, _targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state, data, targets, batch_no + 1)


@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, 0)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 100, update_step_jit, args)
    return params


params = optimization_jit(params, X_batched, y_batched)

print("Training accuracy: ", acc(params, X_train, y_train))
print("Testing accuracy: ", acc(params, X_test, y_test))

######################################################################
# In this example, the variational algorithm is having trouble finding a global minimum even after
# hyperparameter tuning. In the following code, we can see how this performance compares to our other
# proposed strategies.
#

######################################################################
# Observable Construction
# ---------------------
#

######################################################################
# We take combinations of outputs of quantum circuits in this post-variational strategy. We generalize
# the idea of taking classical combinations of quantum states to taking the classical combinations of
# quantum observables by combining the Ansatz :math:`U(\theta)` and observable :math:`O` into a single
# parameterized observable :math:`O(\theta)` and replacing this observable with a collection of
# predefined trial observables :math:`O_1, O_2, \ldots , O_m`. Under this setting, measurement results
# on the quantum circuits are then combined classically, where the optimal weights of each measurement
# is computed via feeding our measurements through a classical multilayer perceptron.
#

######################################################################
# Then, we generate a series of :math:`k`-local observables on that we will conduct measurements with.
#


def local_pauli_group(qubits: int, locality: int):
    assert locality <= qubits, f"Locality must not exceed the number of qubits."
    return list(generate_paulis(0, 0, "", qubits, locality))


def generate_paulis(identities: int, paulis: int, output: str, qubits: int, locality: int):
    if len(output) == qubits:
        yield output
    else:
        yield from generate_paulis(identities + 1, paulis, output + "I", qubits, locality)
        if paulis < locality:
            yield from generate_paulis(identities, paulis + 1, output + "X", qubits, locality)
            yield from generate_paulis(identities, paulis + 1, output + "Y", qubits, locality)
            yield from generate_paulis(identities, paulis + 1, output + "Z", qubits, locality)


######################################################################
# For each image sample, we measure the output of the quantum circuit using the k-local observables
# sequence, and perform logistic regression on these outputs. We do this for 1-local, 2-local and
# 3-local in the for-loop below.
#

train_accuracies_O = []
test_accuracies_O = []
for locality in range(1, 4):
    print(str(locality) + "-local: ")
    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev)
    def circuit(features):
        measurements = local_pauli_group(8, locality)
        feature_map(features)
        return [
            qml.expval(qml.pauli.string_to_pauli_word(measurement)) for measurement in measurements
        ]

    vcircuit = jax.vmap(circuit)
    new_X_train = np.asarray(vcircuit(jnp.array(X_train))).T
    new_X_test = np.asarray(vcircuit(jnp.array(X_test))).T
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50).fit(new_X_train, y_train)
    print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
    print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
    acc = clf.score(new_X_train, y_train)
    train_accuracies_O.append(acc)
    print("Training accuracy: ", acc)
    acc = clf.score(new_X_test, y_test)
    test_accuracies_O.append(acc)
    print("Testing accuracy: ", acc)
    print()

locality = ("1-local", "2-local", "3-local")
train_accuracies_O = [round(value, 2) for value in train_accuracies_O]
test_accuracies_O = [round(value, 2) for value in test_accuracies_O]
x = np.arange(3)
width = 0.25
fig, ax = plt.subplots(layout="constrained")
rects = ax.bar(x, train_accuracies_O, width, label="Training", color="#FF87EB")
ax.bar_label(rects, padding=3)
rects = ax.bar(x + width, test_accuracies_O, width, label="Testing", color="#70CEFF")
ax.bar_label(rects, padding=3)
ax.set_xlabel("Locality")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of different localities")
ax.set_xticks(x + width / 2, locality)
ax.legend(loc="upper left", ncols=3)
plt.show()

######################################################################
# We can see that the highest accuracy is achieved with the 3-local observables, which gives the model
# more information about the outputs of the circuit. However, this is much more computationally
# resource heavy than its lower-locality counterparts.
#

######################################################################
# Ansatz Expansion
# ---------------------
#

######################################################################
# We can also begin with a variational algorithm and replace the parameterized Ansatz U(θ) with an
# ensemble of parameterised fixed Ansatze, by subbing our pre-determined parameters into the rotation
# gates in the ansatz:
#

######################################################################
# The following code is used to generate a series of fixed parameters that would be encoded into the
# ansatz.
#

import numpy as np
from itertools import combinations


def deriv_params(thetas: int, order: int):
    def generate_shifts(thetas: int, order: int):
        shift_pos = list(combinations(np.arange(thetas), order))
        params = np.zeros((len(shift_pos), 2 ** order, thetas))
        for i in range(len(shift_pos)):
            for j in range(2 ** order):
                for k, l in enumerate(f"{j:0{order}b}"):
                    if int(l) > 0:
                        params[i][j][shift_pos[i][k]] += 1
                    else:
                        params[i][j][shift_pos[i][k]] -= 1
        params = np.reshape(params, (-1, thetas))
        return params

    param_list = [np.zeros((1, thetas))]
    for i in range(1, order + 1):
        param_list.append(generate_shifts(thetas, i))
    params = np.concatenate(param_list, axis=0)
    params *= np.pi / 2
    return params


######################################################################
# We construct the ansatz above and measure the top qubit with Pauli-Z.
#

n_wires = 8
dev = qml.device("default.qubit", wires=n_wires)


@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(features, params, n_wires=8):
    feature_map(features)
    ansatz(params)
    return qml.expval(qml.PauliZ(0))


######################################################################
# For each image sample, we measure the outputs of each parameterised circuit for each feature, and
# perform logistic regression on these outputs.
#

train_accuracies_AE = []
test_accuracies_AE = []
for order in range(1, 4):
    print("Order number: " + str(order))
    to_measure = deriv_params(16, order)

    new_X_train = []
    for thing in X_train:
        result = circuit(thing, to_measure.T)
        new_X_train.append(result)
    new_X_test = []
    for thing in X_test:
        result = circuit(thing, to_measure.T)
        new_X_test.append(result)
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50).fit(new_X_train, y_train)
    print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
    print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
    acc = clf.score(new_X_train, y_train)
    train_accuracies_AE.append(acc)
    print("Training accuracy: ", acc)
    acc = clf.score(new_X_test, y_test)
    test_accuracies_AE.append(acc)
    print("Testing accuracy: ", acc)
    print()

######################################################################
# We can see that higher orders give higher testing accuracy. However, it is also more computationally
# expensive due to the number of parameters required as shown by the number of parameters in each
# order below.
#

print("1st order: " + str(deriv_params(16, 1).shape[0]))
print("2nd order: " + str(deriv_params(16, 2).shape[0]))
print("3rd order: " + str(deriv_params(16, 3).shape[0]))

locality = ("1-order", "2-order", "3-order")
train_accuracies_AE = [round(value, 2) for value in train_accuracies_AE]
test_accuracies_AE = [round(value, 2) for value in test_accuracies_AE]
x = np.arange(3)
width = 0.25
fig, ax = plt.subplots(layout="constrained")
rects = ax.bar(x, train_accuracies_AE, width, label="Training", color="#FF87EB")
ax.bar_label(rects, padding=3)
rects = ax.bar(x + width, test_accuracies_AE, width, label="Testing", color="#70CEFF")
ax.bar_label(rects, padding=3)
ax.set_xlabel("Order")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of different derivative orders")
ax.set_xticks(x + width / 2, locality)
ax.legend(loc="upper left", ncols=3)
plt.show()

######################################################################
# Hybrid Strategy
# ---------------------
#

# When taking the strategy of observable construction, one additionally may want to use Ansatz quantum
# circuits to increase the complexity of the model. Hence, we discuss a simple hybrid strategy that
# combines both the usage of Ansatz expansion and observable construction. For each feature, we may
# first expand the ansatz with each of our parameters, then use each k-local observable to conduct
# measurements.
#
# Due to the high number of circuits needed to be computed in this strategy, one may choose to conduct
# the pruning mentioned in our paper, but this is not conducted in this demo.
#
# Note that in our example, we have only tested 3 hybrid samples to reduce the running time of this
# script, but one may choose to try other combinations of the 2 strategies to potentially obtain
# better results.
#

train_accuracies = np.zeros([4, 4])
test_accuracies = np.zeros([4, 4])

for order in range(1, 4):
    for locality in range(1, 4):
        if locality + order > 3 or locality + order == 0:
            continue
        print("Locality: " + str(locality) + " Order: " + str(order))

        dev = qml.device("default.qubit", wires=8)
        params = deriv_params(16, order).T

        @qml.qnode(dev)
        def circuit(features, params):
            measurements = local_pauli_group(8, locality)
            feature_map(features)
            ansatz(params)
            return [
                qml.expval(qml.pauli.string_to_pauli_word(measurement))
                for measurement in measurements
            ]

        vcircuit = jax.vmap(circuit)
        new_X_train = np.asarray(
            vcircuit(jnp.array(X_train), jnp.array([params for i in range(len(X_train))]))
        )
        new_X_train = np.moveaxis(new_X_train, 0, -1).reshape(
            -1, len(local_pauli_group(8, locality)) * len(deriv_params(16, order))
        )
        new_X_test = np.asarray(
            vcircuit(jnp.array(X_test), jnp.array([params for i in range(len(X_test))]))
        )
        new_X_test = np.moveaxis(new_X_test, 0, -1).reshape(
            -1, len(local_pauli_group(8, locality)) * len(deriv_params(16, order))
        )
        clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50).fit(new_X_train, y_train)
        print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
        print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
        acc = clf.score(new_X_train, y_train)
        train_accuracies[locality][order] = acc
        print("Training accuracy: ", acc)
        acc = clf.score(new_X_test, y_test)
        test_accuracies[locality][order] = acc
        print("Testing accuracy: ", acc)
        print()

######################################################################
# Upon obtaining our hybrid results, we may now combine these results with that of the observable
# construction and ansatz expansion menthods, and plot all the post-variational strategies together on
# a heatmap.
#

for locality in range(1, 4):
    train_accuracies[locality][0] = train_accuracies_O[locality - 1]
    test_accuracies[locality][0] = test_accuracies_O[locality - 1]
for order in range(1, 4):
    train_accuracies[0][order] = train_accuracies_AE[order - 1]
    test_accuracies[0][order] = test_accuracies_AE[order - 1]

import matplotlib.colors

cvals = [0, 0.5, 0.85, 0.95, 1]
colors = ["black", "#C756B2", "#FF87EB", "#ACE3FF", "#D5F0FD"]
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


locality = ["top qubit\n Pauli-Z", "1-local", "2-local", "3-local"]
order = ["0th Order", "1st Order", "2nd Order", "3rd Order"]

fig, ax = plt.subplots()
im = ax.imshow(train_accuracies, cmap=cmap, origin="lower")

ax.set_yticks(np.arange(len(locality)), labels=locality)
ax.set_xticks(np.arange(len(order)), labels=order)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(locality)):
    for j in range(len(order)):
        text = ax.text(
            j, i, round(train_accuracies[i, j], 2), ha="center", va="center", color="black"
        )

ax.set_title("Training Accuracies")
fig.tight_layout()
plt.show()

locality = ["top qubit\n Pauli-Z", "1-local", "2-local", "3-local"]
order = ["0th Order", "1st Order", "2nd Order", "3rd Order"]

fig, ax = plt.subplots()
im = ax.imshow(test_accuracies, cmap=cmap, origin="lower")

ax.set_yticks(np.arange(len(locality)), labels=locality)
ax.set_xticks(np.arange(len(order)), labels=order)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(locality)):
    for j in range(len(order)):
        text = ax.text(
            j, i, round(test_accuracies[i, j], 2), ha="center", va="center", color="black"
        )

ax.set_title("Test Accuracies")
fig.tight_layout()
plt.show()


######################################################################
# Experimental Results
# ====================
#

######################################################################
# Our results show that all post-variational methods exceed the variational algorithm while using the
# same Ansatz for the Ansatz expansion and hybrid strategies.
#
# However, given that the post-variational algorithms extracts more features than the classical
# algorithm, there are more parameters to optimize, leading to overfitting on the training model to a
# certain extent, as shown by the decreasing testing accuracy of these models.
#
# From these performance results, we can obtain a glimpse of the effectiveness of each strategy. While
# the observable construction strategy does not perform much better even when we use 3-local
# observables, the inclusion of 1-local and 2-local observables provide a boost in accuracy when used
# in conjunction with first order derivatives in the hybrid strategy. This implies that the addition
# of the observable expansion strategy can serve as an heuristic to expand the expressibility to
# ansatz expansion method but may not be sufficient in itself as a good training strategy.
#

######################################################################
# Conclusion
# ---------------------
#

######################################################################
# In this tutorial, we have implemented the post variational strategies to classify handwritten digits
# of threes and fives.
#
# Comparing to variational algorithms, we note that by using our heuristic strategies, we can also
# potentially lower the number of quantum gates per quantum circuit. By replacing part of the Ansatz
# with an ensemble of local Pauli measurements as with our observable construction method, one reduces
# the depth of the circuit. Using the Ansatz expansion strategy results in fixed circuits. These fixed
# circuits we can optimize with transpilation and circuit optimization strategies.
#
# While our empirical results show that there are cases where the usage of post-variational quantum
# neural net- works surpass the performance of variational algorithm, we do not make a statement on
# the superiority of variational and post-variational algorithms as different problem settings may
# lead to different algorithms outperforming the other. We propose post-variational quantum neural
# networks simply as an alternative implementation of neural networks in the NISQ setting, and leave
# the determination of case-by-case distinctions on performance evaluations and resource consumption
# to future work.
#

######################################################################
# Based on: https://arxiv.org/pdf/2307.10560 [#huang2024postvariational]_
#
#
# References
# ~~~~~~~~~~
#
# .. [#cerezo2021variational]
#
#     M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, 
#     J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, 
#     Variational quantum algorithms, 
#     `Nat. Rev. Phys. 3, 625, (2021) <https://doi.org/10.1038/s42254-021-00348-9>`__.
#
#
# .. [#schuld2019evaluating]
#
#     M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, 
#     Evaluating analytic gradients on quantum hardware,
#     `Phys. Rev. A. 99, 032331, (2019) <https://doi.org/10.1103/PhysRevA.99.032331>`__. 
#     
#
# .. [#mcclean2018barren]
#
#     J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven,
#     Barren plateaus in quantum neural network training landscapes, 
#     `Nat. Commun. 9, 4812, (2018) <https://doi.org/10.1038/s41467-018-07090-4>`__.
#
#
# .. [#huang2024postvariational]
#
#     P.-W. Huang and P. Rebentrost, 
#     Post-variational quantum neural networks (2024), 
#     `arXiv:2307.10560 [quant-ph] <https://arxiv.org/abs/2307.10560>`__.
#
#
# .. [#du2020expressive]
#
#     Y. Du, M.-H. Hsieh, T. Liu, and D. Tao,  
#     Expressive power of parametrized quantum circuits,  
#     `Phys. Rev. Res. 2, 033125 (2020) <https://doi.org/10.1103/PhysRevResearch.2.033125>`__.
#
#

##############################################################################
# About the authors
# ---------------------
#
