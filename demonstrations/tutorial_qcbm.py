"""
.. qcbm:

QCBM with Tensor Networks Ansatz
===============


In this tutorial we employ the NISQ-friendly generative model known as the Quantum Circuit Born Machine (QCBM) introduced by
Benedetti, Garcia-Pintos, Perdomo, Leyton-Ortega, Nam and Perdomo-Ortiz (2019) in [#Benedetti]_, to obtain the probability
distribution of the bars and stripes data set. To this end, we use the tensor-network inspired templates available in Pennylane
to construct the model's ansatz.
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
# An important trait of the generative model that allows for this generation of new samples is rooted in its
# ability to capture correlations present in the input data. Among the different models employed in the literature
# for generative machine learning, the so-called "Born machine" model is based on representing the probability
# distribution :math:`p(x)` in terms of the amplitudes of the quantum wavefunction :math:`\ket{\psi}`, with the
# relation given by Born's rule
#
# .. math::
#   p(x) = \lVert\bra{x}\ket{\psi}\rVert^2
#
# As done in [#Cheng]_, the efficient representation provided by tensor network ansätze invites us to represent
# the wavefunction in terms of tensor networks classes. In particular, the ubiquitous classes of Matrix Product
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
# from tensor networks efficiently ([#Ferris]_, [#Ballarin]_), this task appears as a suitable candidate to attempt and
# achieve quantum advantage in Noisy Intermediate-Scale Quantum (NISQ) devices, as suggested in [#Harrow]_.
# As presented in [#Benedetti]_, this approach employing quantum circuits to model the probabililty distribution is
# known as the Quantum Circuit Born Machine (QCBM).
#
# The problem formulation starts by looking at the training dataset
# :math:`\mathfrak{D} = (\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(D)})` made up of :math:`D` binary
# vectors of length :math:`N`. Each of these vectors has an associated probability of occurring within the data,
# resulting in the target probability distribution :math:`P_{\mathfrak{D}}`. For a quantum circuit
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
# Then, we can look at the probability of finding the output wavefunction in the computational basis state
# :math:`\ket{\mathbf{x}}` expressed as
#
# .. math::
#   P_\theta(\mathbf{x}) = \lVert\bra{\mathbf{x}}\ket{\psi(\theta)}\rVert^2
#
# We can then use this quantity to define a cost function to be minimized by iteratively optimizing the parameter
# vector :math:`\theta` in order to obtain the wavefunction that best approximates the target distribution.
# In other words, we can formulate this problem as the minimization of
#
# .. math::
#   \min_{\theta} C(P_\theta(\mathbf{x})),
#
# where :math:`C` is the cost function to be optimized, which can take different forms based on the specific
# scenario, with the common denominator being that all of them should quantify the difference between the target
# probability and the probability distribution obtained from the quantum circuit. Due to the nature of the finite
# sampling used to estimate the distribution, analogous to what is done in classsical generative machine
# learning [#Goodfellow]_, in this tutorial we choose the cost function to be the Kullback-Leibler (KL) divergence:
#
# .. math::
#   C(\theta) = \sum_{\mathbf{x}} P_\mathfrak{D}(\mathbf{x}) \ln \left ( \frac{P_D(\mathbf{x})}{P_\theta(\mathbf{x})} \right)
#
#
# Tensor Network Ansatz
# ---------------------------
# The algorithm presented in [#Benedetti]_ proposes the use of a hardware-efficient ansatz to prepare
# the probability distribution using a quantum circuit. However, in this work we take inspiration from previous
# approaches to generative modelling using tensor networks to represent the wavefunction, as done in [#Han]_
# and [#Cheng]_ employing MPS and TTN, respectively. Since quantum circuits are a restricted class of tensor
# networks, there is a natural relation between them that we can exploit to define a tensor-network inspired ansatz.
# In particular, we take into consideration the local correlations of the data, and employ the MPS and
# TTN circuit ansatz implemented in Pennylane. See `this tutorial <https://pennylane.ai/qml/demos/tutorial_tn_circuits/>`_
# for a clearer understanding of these ansätze. The conversion between a TTN of bond dimension 2 into a quantum circuit looks as
# follows.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/ttn_ansatz.jpg
#   :align: center
#   :width: 70 %
#
# Analagously, converting a bond dimension 2 MPS tensor network into a quantum circuit would be done as shown in the following figure.
#
# .. figure:: ../_static/demonstration_assets/qcbm_tensor_network/mps_ansatz.jpg
#   :align: center
#   :width: 70 %
#
#
# Pennylane implementation
# ---------------------------
# For this demo, we need a bit of an extra help from `another repo <https://github.com/XanaduAI/qml-benchmarks>`_ that
# will allow us to generate the dataset used for training, i.e. the bars and stripes data set.
#
# .. code-block:: bash
#
#    git clone https://github.com/XanaduAI/qml-benchmarks.git
#    cd qml-benchmarks
#    pip install .
#
# With that out of the way, we can start by importing what we need

from typing import Literal
import pennylane as qml
from functools import partial
from qml_benchmarks.data.bars_and_stripes import generate_bars_and_stripes
import matplotlib.pyplot as plt
import jax.numpy as np
import jax
import optax

##############################################################################
# We then configure our random seed and setup the constants that we will need for our experiments

dev = qml.device("default.qubit", wires=16)
key = jax.random.PRNGKey(1)

QUBITS = 16
DATASET_SIZE = 1000
TRAINING_ITERATIONS = 100
N_PARAMS_PER_BLOCK = 15
LEARNING_RATE = 0.01


MPS_DATA_SHAPE = (qml.MPS.get_n_blocks(range(QUBITS), 2), N_PARAMS_PER_BLOCK)
TTN_DATA_SHAPE = (2 ** int(np.log2(QUBITS / 2)) * 2 - 1, N_PARAMS_PER_BLOCK)


##############################################################################
# For generating our dataset, which in our case is the bars and stripes dataset, we declare the following function.
def prepare_dataset(size, rows, cols):
    X, _ = generate_bars_and_stripes(size, rows, cols, noise_std=0.0)

    # We need to transform the dataset from the {-1, 1}
    # to the binary {0, 1} range for easier processing
    X += 1
    X //= 2

    # Then we need to flatten the dataset
    X = X.squeeze().reshape(X.shape[0], -1)

    unique_elems, true_probs = np.unique(X, return_counts=True, axis=0)
    idxs = unique_elems.dot(1 << np.arange(unique_elems.shape[-1] - 1, -1, -1)).astype(np.int32)

    return idxs, true_probs / size


##############################################################################
# The last part of the function calculates the empirical probabilities of the generated samples
# from the helper function `generate_bars_and_stripes`. Note that we also need the integer value
# of the generated samples to use as an index for the probability vectors that are going to be
# extracted from our circuits.
#
# Next, we build our circuits for both the MPS and TTN ansätze. For the main building block of each node
# in the ansatz, we are going to be using an SU(4) block, since it can express any 2-qubit unitary.


def su4_block(weights, wires):
    qml.Rot(weights[0], weights[1], weights[2], wires=wires[0])
    qml.Rot(weights[3], weights[4], weights[5], wires=wires[1])

    qml.CNOT(wires=wires)

    qml.RZ(weights[6], wires=wires[1])
    qml.RY(weights[7], wires=wires[1])
    qml.RX(weights[8], wires=wires[1])

    qml.CNOT(wires=wires)

    qml.Rot(weights[9], weights[10], weights[11], wires=wires[0])
    qml.Rot(weights[12], weights[13], weights[14], wires=wires[1])


@partial(jax.jit, static_argnums=(1,))
@qml.qnode(dev, interface="jax")
def qcbm_circuit(template_weights, template_type: Literal["MPS", "TTN"]):
    tn_ansatz = getattr(qml, template_type)(
        wires=range(QUBITS),
        n_block_wires=2,
        block=su4_block,
        n_params_block=N_PARAMS_PER_BLOCK,
        template_weights=template_weights,
    )

    if template_type == "TTN":
        qml.adjoint(tn_ansatz)

    return qml.probs()


##############################################################################
# You will notice that we do not need to embed any data into the circuit, since QCBMs are generative models!
# Any interaction with our dataset will take place in the cost function.
#
# You can also see how we are applying the adjoing operation on the template when using a TTN.
# This is a minor hack to mirror the ansatz. This is needed since for generation, we want the root
# of the tree to be at the beginning of the circuit, as described in [#Stoudenmire]_.
#
# One final thing to notice is the number of qubits each block will be affecting. Our choice of 2 here
# reflects the expressivity of the model, since it is directly correlated to the bond dimensions of
# the tensor networks as illustrated in the previous section. For larger, more expressive bond dimensions,
# gates interacting with more qubits is needed, but that would not necessarily be hardware-efficient on the current NISQ devices.
#
# Now, we define our loss, which, as mentioned before, will be the KL-Divergence loss:


def kl_div(p, q):
    return np.sum(q * np.log(q / p))


##############################################################################
# We are now ready to train some circuits! Let's generate our dataset first and prepare the training function.
idxs, true_probs = prepare_dataset(DATASET_SIZE, 4, 4)
optimizer = optax.adam(LEARNING_RATE)


def train_circuit(cost_fn, init_weights, optimizer):
    opt_state = optimizer.init(init_weights)
    weights = init_weights

    costs = []
    for it in range(TRAINING_ITERATIONS + 1):
        grads = jax.grad(cost_fn)(weights, idxs=idxs, true_probs=true_probs)
        updates, opt_state = optimizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)

        current_cost = cost_fn(weights, idxs, true_probs)

        costs.append(current_cost)

        if it % 10 == 0:
            print(f"Iter: {it:4d} | KL Loss: {current_cost:0.7f}")

    return costs, weights


##############################################################################
# Let's start with the MPS circuit:
def cost_fn_mps(weights, idxs, true_probs):
    probs = qcbm_circuit(weights, template_type="MPS")
    pred_probs = probs[idxs]
    return kl_div(pred_probs, true_probs)


mps_weights_init = jax.random.uniform(key=key, shape=MPS_DATA_SHAPE)

mps_costs, mps_weights = train_circuit(cost_fn_mps, mps_weights_init, optimizer)

##############################################################################
# Now we do the same for the TTN network, resetting the optimizer first.
optimizer = optax.adam(LEARNING_RATE)


def cost_fn_ttn(weights, idxs, true_probs):
    probs = qcbm_circuit(weights, template_type="TTN")
    pred_probs = probs[idxs]
    return kl_div(pred_probs, true_probs)


ttn_weights_init = jax.random.uniform(key=key, shape=TTN_DATA_SHAPE)

ttn_costs, ttn_weights = train_circuit(cost_fn_ttn, ttn_weights_init, optimizer)

##############################################################################
# We can plot the loss curves of both models and see how their performance differs

plt.plot(range(TRAINING_ITERATIONS + 1), mps_costs, "-.b", label="MPS Losses")
plt.plot(range(TRAINING_ITERATIONS + 1), ttn_costs, "-.g", label="TTN Losses")
plt.xlabel("Iteration #", fontsize=16)
plt.ylabel("KL Loss", fontsize=16)
plt.legend()
plt.show()

##############################################################################
# As expected, the TTN ansatz is able to achieve a smaller loss compared to the MPS one.
# This is due to the topology of the TTN ansatz able to capture non-adjacent correlations
# in the data thanks to its tree structure.
# Let's now generate some samples from our models and see how stripe-y they are:


def generate_and_plot(circuit_fn, weights, template):
    probs = circuit_fn(weights, template)
    generated_samples = jax.random.choice(key=key, a=len(probs), shape=(9,), p=probs)
    generated_samples_bin = (
        (generated_samples.reshape(-1, 1) & (2 ** np.arange(QUBITS))) != 0
    ).astype(int)

    plt.figure(figsize=[3, 3])
    for i, sample in enumerate(generated_samples_bin):
        plt.subplot(3, 3, (i % 9) + 1)
        plt.imshow(np.reshape(sample[::-1], [4, 4]), cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.show()


##############################################################################
# First, we generate samples from the MPS model
generate_and_plot(qcbm_circuit, mps_weights, "MPS")

##############################################################################
# Then, we generate samples from the TTN model
generate_and_plot(qcbm_circuit, ttn_weights, "TTN")

##############################################################################
# We can see that for both models, the generated samples are not perfect, which is
# a sign that our models are not expressive enough for the task at hand. As explained before,
# one can play around with blocks affecting more qubits at once which would allow the models
# to capture the more minute correlations and create better entanglement in the model.
# One can also experiment with more standard topologies such as the `MERA ansatz <https://docs.pennylane.ai/en/stable/code/api/pennylane.MERA.html>`_
# or the PEPS topology, which is very suitable for 2D datasets such as ours.

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
#    "Optimal sampling of tensor networks targeting wavefunction's fast decaying tails",
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
# .. [#Stoudenmire]
#    Huggins, William and Patil, Piyush and Mitchell, Bradley and Whaley, K Birgitta and Stoudenmire, E Miles
#    "Towards quantum machine learning with tensor networks",
#    `<https://arxiv.org/abs/1803.11537>`__, 2018.
#
# About the authors
# ----------------
