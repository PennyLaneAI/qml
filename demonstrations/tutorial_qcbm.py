"""
.. qcbm:

QCBM with Tensor Networks Ansatz
===============


In this tutorial we employ the NISQ-friendly generative model known as the Quantum Circuit Born Machine (QCBM) introduced by Benedetti, Garcia-Pintos, Perdomo, Leyton-Ortega, Nam and Perdomo-Ortiz (2019) in [#Benedetti]_. to obtain the probability distribution of the bars and stripes data set. To this end, we use the tensor-network inspired templates available in Pennylane to construct the model's ansatz.
"""

##############################################################################
# Generative modeling and quantum physics
# -----------------------
# In the classical setup of unsupervised learning, a generative statistical model is tasked with creating
# new data instances that follow the probability distribution of the input training data. This is in contrast
# to other statistical models like the discriminative model, which allows us to "tell apart" the different instances
# present in the the input data, assigning a label to each of them.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/generative_discriminative.jpg
#   :align: center
#   :height: 300
#
# An important trait of the generative model that allows for this generation of new samples is rooted on its
# ability to capture correlations present in the input data. Among the different models employed in the literature
# for generative machine learning, the so called "Born machine" model is based on representing the probability
# distribution :math:`p(x)` in terms of the amplitudes of the quantum wave function :math:`\ket{\psi}`, with the
# relation given by Born's rule
#
# .. math::
#   p(x) = \lVert\bra{x}\ket{\psi}\rVert^2
#
# As done in [#Cheng]_ the efficient representation provided by tensor network ansätze invites us to represent
# the wave function in terms of tensor networks classes. In particular, the ubiquitous classes of Matrix Product
# States (MPS) and Tree Tensor Networks (TTN) are capable of capturing the local correlations present in the
# training data, thus making them suitable candidates be employed in the generative model.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/tensor_network_wavefunction.jpg
#   :align: center
#   :height: 300
#
# Tensor networks and Quantum Circuits
# -----------------------
# As part of the generative model pipeline, it is necessary to draw finite samples from the distribution represented by the wavefunction
# in order to approximate the target probability distribution. While many algorithms have been proposed to sample
# tensor networks classes efficiently ([#Ferris]_, [#Ballarin]_), as suggested in [#Harrow]_, this task appears as
# a suitable candidate to attempt and achieve quantum advantage in Near Intermediate Scale Quantum (NISQ) devices.
# As presented in [#Benedetti]_, this approach employing quantum circuits to model the probabililty distribution is
# known as the Quantum Circuit Born Machine (QCBM).
#
# The problem formulation starts by looking at the training dataset
# :math:`\mathcal{D} = (\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(D)})` made up of :math:`D` binary
# vectors of length :math:`N`. Each of these vectors has an associated probability of occurring within the data,
# resulting in the target probability distribution :math:`P_{\mathcal{D}}`. For a quantum circuit
# with :math:`N` qubits, this formulation gives rise to the one-to-one mapping betweeen the computational
# states and the input vectors
#
# .. math::
#   \mathbf{x} := (x_1, x_2, \ldots, x_N) \leftrightarrow \ket{\mathbf{x}} := \ket{x_1, x_2, \ldots, x_N}
#
# To approximate the target distribution, we can create an ansatz for the quantum circuit parametrized by a
# vector :math:`\theta`, such that the output state vector is defined as :math:`\ket{\psi(\theta)} = U(\theta) \ket{0}`.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/quantum_circuit.jpg
#   :align: center
#   :width: 50%
#
# Then, we can look at the probability of finding the output wave function in the computational basis state
# :math:`\ket{\mathbf{x}}` expressed as
#
# .. math::
#   P_\theta(\mathbf{x}) = \lVert\bra{\mathbf{x}}\ket{\psi(\theta)}\rVert^2
#
# We can then use this quantity to define a cost function to be minimized by iteratively optimizing the parameter
# vector :math:`\theta` in order to obtain the wave function that best approximates the target distribution.
# In other words, we can formulate this problem as the minimization
#
# .. math::
#   \min_{\theta} C(P_\theta(\mathbf{x})),
#
# where :math:`C` is the cost function to be optimized, which can take different forms based on the specific
# scenario, with the common denominator being that all of them should quantify the difference between the target
# probability and the probability distribution obtained from the quantum circuit. Due to the nature of the finite
# sampling used to estimate the distribution, analogous to what is done in classsical generative machine
# learning [#Goodfellow]_, in this tutorial we choose the cost function to be the Kullback-Leibler (KL) divergence.
#
# .. math::
#   C(\theta) = \sum_{\mathbf{x}} P_D(\mathbf{x}) \ln \left ( \frac{P_D(\mathbf{x})}{P_\theta(\mathbf{x})} \right)
#
#
# Tensor Network Ansatz
# ---------------------------
# The algorithm presented in [#Benedetti]_ proposes the use of a hardware-efficient ansatz to prepare
# the probability distribution using a quantum circuit. However, in this work we take inspiration from previous
# approaches to generative modelling using tensor networks to represent the wave function, as done in [#Han]_
# and [#Cheng]_ employing MPS and TTN, respectively. Since quantum circuits are a restricted class of tensor
# networks, there is a natural relation between them that we can exploit to define a tensor-network inspired ansatz.
# In particular, we take into consideration the local correlations of the data, and employ the MPS and
# TTN circuit ansatz implemented in Pennylane. See this tutorial `this tutorial <https://pennylane.ai/qml/demos/tutorial_tn_circuits/>`_
# for a deeper study of these ansätze. The conversion between the TTN class into a quantum circuit looks as
# follows.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/ttn_ansatz.jpg
#   :align: center
#   :width: 70 %
#
# Analagously, converting an MPS tensor network into a quantum circuit would be done as shown in the following figure.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/mps_ansatz.jpg
#   :align: center
#   :width: 70 %
#
#
# Pennylane implementation
# ---------------------------
# For this demo, we need a bit of an extra help from `another repo <https://github.com/XanaduAI/qml-benchmarks>`_ that
# will allow us to generate the dataset needed in this tutorial:
#
# .. code-block:: bash
#
#    git clone https://github.com/XanaduAI/qml-benchmarks.git
#    cd qml-benchmarks
#    pip install .
#
# With that out of the way, we can start by importing what we need

import pennylane as qml
from pennylane import numpy as np
from qml_benchmarks.data.bars_and_stripes import generate_bars_and_stripes
import matplotlib.pyplot as plt

##############################################################################
# We then configure our random seed and setup some global variables that we will need for our experiments

np.random.seed(1)
dev = qml.device("default.qubit", wires=16)

QUBITS = 16
DATASET_SIZE = 1000
TRAINING_ITERATIONS = 50

MPS_DATA_SHAPE = (qml.MPS.get_n_blocks(range(QUBITS), 2), 6)
TTN_DATA_SHAPE = (2 ** int(np.log2(QUBITS / 2)) * 2 - 1, 6)


##############################################################################
# For generating our dataset, which in our case is the Bars and Stripes dataset, we declare the following function:
def prepare_dataset(size, rows, cols):
    X, _ = generate_bars_and_stripes(size, rows, cols, noise_std=0.0)

    # We need to transform the dataset from the {-1, 1} realm
    # to the binary {0, 1} for easier processing
    X += 1
    X //= 2

    # Then we need to flatten the dataset
    X = X.squeeze().reshape(X.shape[0], -1)

    # Finally, we need to determine the integer value
    # of the instances and calculate their empirical probabilities
    # The integer representation will allow for easy access of the
    # required probabilities from the simulation
    unique_elems, true_probs = np.unique(X, return_counts=True, axis=0)
    idxs = unique_elems.dot(1 << np.arange(unique_elems.shape[-1] - 1, -1, -1)).astype(np.int32)

    return idxs, true_probs / size


##############################################################################
# Note that for this demo, we assume that the underlying true probability of the dataset
# (which is a uniform probability for all elements) is unknown. Rather, we calculate an empirical probability from
# the samples generated by the `generate_bars_and_stripes` function.
#
# Next, we build our circuits for both the MPS and TTN ansätze:


def block(weights, wires):
    qml.Rot(weights[0], weights[1], weights[2], wires=wires[0])
    qml.Rot(weights[3], weights[4], weights[5], wires=wires[1])

    qml.CNOT(wires=wires)


@qml.qnode(dev)
def qcbm_circuit_mps(template_weights):
    qml.MPS(
        wires=range(QUBITS),
        n_block_wires=2,
        block=block,
        n_params_block=6,
        template_weights=template_weights,
    )

    return qml.probs()


@qml.qnode(dev)
def qcbm_circuit_ttn(template_weights):
    # We need an adjoint here to reflect the circuit,
    # such that the leaves of the TTN are at the measurement,
    # more faithfully mimicking a TTN.
    qml.adjoint(
        qml.TTN(
            wires=range(QUBITS),
            n_block_wires=2,
            block=block,
            n_params_block=6,
            template_weights=template_weights,
        )
    )

    return qml.probs()


##############################################################################
# You will notice that we do not need to embed any data into the circuit, since QCBMs are generative models!
# Any interaction with our dataset will take place in the cost function.
# Now, we define our loss, which, as mentioned before, will be the KL-Divergence loss:


def kl_div(p, q):
    return np.sum(q * np.log(q / p))


##############################################################################
# We are now ready to train some circuits! Let's generate our dataset first and prepare the optimizer.
idxs, true_probs = prepare_dataset(DATASET_SIZE, 4, 4)
optimizer = qml.AdamOptimizer()


##############################################################################
# Let's start with the MPS circuit:
def cost_fn_mps(weights, idxs, true_probs):
    probs = qcbm_circuit_mps(weights)
    pred_probs = probs[idxs]
    return kl_div(pred_probs, true_probs)


mps_weights_init = np.random.random(size=MPS_DATA_SHAPE)
mps_weights = mps_weights_init

mps_costs = []
for it in range(TRAINING_ITERATIONS + 1):
    mps_weights = optimizer.step(cost_fn_mps, mps_weights, idxs=idxs, true_probs=true_probs)

    current_cost = cost_fn_mps(mps_weights, idxs, true_probs)

    mps_costs.append(current_cost)

    if it % 10 == 0:
        print(f"Iter: {it:4d} | KL Loss: {current_cost:0.7f}")

##############################################################################
# Now we do the same for the TTN network, resetting the optimizer first.
optimizer.reset()


def cost_fn_ttn(weights, idxs, true_probs):
    probs = qcbm_circuit_ttn(weights)
    pred_probs = probs[idxs]
    return kl_div(pred_probs, true_probs)


ttn_weights_init = np.random.random(size=TTN_DATA_SHAPE)
ttn_weights = ttn_weights_init

ttn_costs = []
for it in range(TRAINING_ITERATIONS + 1):
    ttn_weights = optimizer.step(cost_fn_ttn, ttn_weights, idxs=idxs, true_probs=true_probs)

    current_cost = cost_fn_ttn(ttn_weights, idxs, true_probs)

    ttn_costs.append(current_cost)

    if it % 10 == 0:
        print(f"Iter: {it:4d} | KL Loss: {current_cost:0.7f}")

##############################################################################
# Let's now plot the loss curves of the experiments against each other

plt.plot(range(TRAINING_ITERATIONS + 1), mps_costs, ".b", label="MPS Losses")
plt.plot(range(TRAINING_ITERATIONS + 1), ttn_costs, ".g", label="TTN Losses")
plt.xlabel("Iteration #", fontsize=16)
plt.ylabel("KL Loss", fontsize=16)
plt.legend()
plt.show()

##############################################################################
# References
# ----------
#
# .. [#Benedetti]
#    M. Benedetti, D. Garcia-Pintos, O. Perdomo, V. Leyton-Ortega, Y. Nam, and A. Perdomo-Ortiz.
#    "A generative modeling approach for benchmarking and training shallow quantum circuits",
#    `<https://arxiv.org/abs/1801.07686>`__, 2019.
#
# .. [#Cheng]
#    Song Cheng, Lei Wang, Tao Xiang, and Pan Zhang. "Tree tensor networks for generative modeling"
#    `<http://dx.doi.org/10.1103/PhysRevB.99.155131>`__, 2019.
#
# .. [#Ferris]
#    Andrew J. Ferris and Guifre Vidal. "Perfect sampling with unitary tensor networks",
#    `<http://dx.doi.org/10.1103/PhysRevB.85.165146>`__, 2020.
#
# .. [#Ballarin]
#    Marco Ballarin, Pietro Silvi, Simone Montangero, and Daniel Jaschke.
#    "Optimal sampling of tensor networks targeting wave function's fast decaying tails",
#    `<https://arxiv.org/abs/2401.10330>`__, 2024.
#
# .. [#Harrow]
#    Aram W. Harrow & Ashley Montanaro. "Quantum computational supremacy",
#    `<http://dx.doi.org/10.1038/nature23458>`__, 2017.
#
# .. [#Goodfellow]
#    Ian Goodfellow, Yoshua Bengio, and Aaron Courville. "Deep Learning",
#    `<http://www.deeplearningbook.org>`__, 2016.
#
# .. [#Han]
#    Z.-Y. Han, J. Wang, H. Fan, L. Wang, and P. Zhang.
#    "Unsupervised Generative Modeling Using Matrix Product States",
#    `<http://dx.doi.org/10.1103/PhysRevX.8.031012>`__, 2018.
#
# About the authors
# ----------------
