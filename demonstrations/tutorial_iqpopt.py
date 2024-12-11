r"""IQPopt: Fast optimization of IQP circuits in JAX

================================================

"""

######################################################################
# The Pennylane library offers the possibility of simulating any type of quantum circuit, but the

# resources needed always scale exponentially with the number of qubits. There are some classes of

# quantum citcuits that we know can be efficiently simulated, and this work attempts to fulfil part of

# this niche.

# 

# IQPopt is a package designed for fast optimization of parameterized Instantaneous Quantum Polynomial

# (IQP) circuits using JAX, resulting in a linear scaling on the number of qubits. These circuits are

# a commonly studied class of quantum circuits comprised of commuting gates, for which we can

# efficiently evaluate expectation values of tensor Pauli operators, but have become well known due to

# a number of results related to the hardness of sampling their output distributions [1-3]. As a

# consequence, IQP circuits are prime candidates for potential quantum advantages for tasks related to

# sampling, and have been the focus of a number of theoretical [4-6] and experimental [7,8] works

# aiming towards this goal.

# 

# In this demo we present the IQPopt package, which contains some tools that allow us to approximate

# expectation values of PauliZ operators for these circuits and, therefore, to evaluate them in

# generative optimization tasks.

# 

######################################################################
# .. figure:: ../_static/demonstration_assets/iqpopt/iqpopt_main.png

#    :alt: IQP circuit optimization

# 

#    IQP circuit optimization

# 

######################################################################
# Creating a circuit

# ------------------

# 

######################################################################
# These are circuits comprised of gates :math:`\text{exp}(i\theta_j X_j)`, where the generator

# :math:`X_j` is a tensor product of Pauli X operators acting on some subset of qubits and

# :math:`\theta_j` is a trainable parameter. Input states and measurements are diagonal in the

# computational (Z) basis.

# 

# To define such a circuit (with input state :math:`\vert 0 \rangle`) we need to specify the number of

# qubits and the parameterized gates

# 

import iqpopt as iqp
from iqpopt.utils import local_gates

n_qubits = 3
gates = local_gates(n_qubits, 2) # all gates with max weight 2

circuit = iqp.IqpSimulator(n_qubits, gates)

######################################################################
# A set of gates is specified by a list

# 

# .. raw:: html

# 

#    <center>

# 

# gates = [gen1, gen2, gen3]

# 

# .. raw:: html

# 

#    </center>

# 

# that contains the generators of the gates. Since all gates have Pauli X type generators, a generator

# is represented by a list that specifies which qubits are acted on by an X operator. For example,

# 

# .. raw:: html

# 

#    <center>

# 

# gen1 = [0,1]

# 

# .. raw:: html

# 

#    </center>

# 

# corresponds to a gate with generator :math:`X\otimes X` acting on the first two qubits of the

# circuit. For a three qubit system the corresponding parameterized gate is therefore

# :math:`\exp(i\theta_1 X\otimes X\otimes \mathbb{I})`.

# 

# The previous example prepares a three qubit circuit with all single and two qubit gates via the

# function local_gates, i.e.

# 

# .. raw:: html

# 

#    <center>

# 

# gates = [[0], [1], [2], [0,1], [0,2], [1,2]]

# 

# .. raw:: html

# 

#    </center>

# 

######################################################################
# Estimating expectation values

# -----------------------------

# 

######################################################################
# The class method ``IqpSimulator.op_expval()`` is used to provide estimates of expectation values.

# The function requires a parameter array ``params``, a PauliZ operator specified by its binary

# representation ``op``, the number of samples ``n_samples`` that controls the precision of the

# approximation (the more the better), and a JAX pseudo random number generator key to seed the

# randomness of the sampling.

# 

# Below is a simple example implementing this for the three qubit circuit defined above, which returns

# the expectation value estimate as well as its standard error.

# 

import jax
import jax.numpy as jnp
import numpy as np

op = jnp.array([0,1,0]) # Pauli Z on the second qubit
params = jnp.array(np.random.rand(len(circuit.gates)))
n_samples = 1000
key = jax.random.PRNGKey(42)

expval, std = circuit.op_expval(params, op, n_samples, key)

print(expval, std)

######################################################################
# The function also allows for fast batch evaluation of expectation values. If we specify a batch of

# operators ``ops`` by an array we can batch evaluate the expectation values and errors in parallel

# with the same syntax.

# 

ops = jnp.array([[1,0,0],[0,1,0],[0,0,1]]) # batch of single qubit Pauli Zs

expvals, stds = circuit.op_expval(params, ops, n_samples, key)

print(expvals, stds)

######################################################################
# Testing the expectation values with Pennylane

# ---------------------------------------------

# 

######################################################################
# When using new software, it is always good practice to check your results if possible. In our case

# we have Pennylane, a well tested library that does this calculations for a small number of qubits.

# On the `package's github <https://github.com/XanaduAI/iqpopt>`__ you can find already made functions

# that calculate the same expectation values as ``.op_expval()``, taking the necessary info from an

# ``IqpSimulator`` object.

# 

# Let's first download these functions so we don't have to build them again:

# 

import requests

req = requests.get("https://raw.githubusercontent.com/XanaduAI/iqpopt/refs/heads/main/tests/pennylane_functions.py?token=GHSAT0AAAAAACYVZBT5V4YNS5U2YD6COCQYZ2YM32Q")

with open("pennylane_functions.py", "wb") as f:
    f.write(req.content)

######################################################################
# Using the same ``params`` and ``op`` as before (but using ``numpy`` arrays instead of

# ``jax.numpy``), let's see the pennylane's result.

# 

from pennylane_functions import penn_op_expval

penn_expval = penn_op_expval(circuit, np.array(params), np.array(op))

print(penn_expval)

######################################################################
# Since the calculation on ``iqpopt``\ 's side is stochastic, the results are not exactly the same,

# but as we can see, they are within std error. You can try increasing ``n_samples`` in order to

# obtain closer approximations.

# 

######################################################################
# Sampling and probabilities

# --------------------------

# 

######################################################################
# We can also view a parameterized IQP circuit as a generative model that generates samples of binary

# vectors according to the distribution :raw-latex:`\begin{align}

# q_{\boldsymbol{\theta}}(\boldsymbol{x}) \equiv q(\boldsymbol{x}\vert\boldsymbol{\theta})=\vert\bra{\boldsymbol{x}}U(\boldsymbol{\theta})\ket{0}\vert^2.

# \end{align}` For low amounts of qubits, we can use Penylane's arsenal and know the output

# probabilities of the circuit as well as sample from it. Note that there is not an efficient

# algorithm to do these, so, for large numbers of qubits, these methods will either take too long or

# not work at all.

# 

# These functions are already implemented in the ``IqpSimulator`` object, so we don't have to download

# them.

# 

sample = circuit.sample(params, shots=1)
print(sample)

probabilities = circuit.probs(params)
print(probabilities)

######################################################################
# The ``.probs()`` method works as it does in pennylane, the returned array of probabilities is in

# lexicographic order.

# 

######################################################################
# Optimizing a circuit

# --------------------

# 

######################################################################
# Circuits can be optimized via a separate ``Trainer`` class. To instantiate a trainer object we first

# define a loss function (also called an objective function), an optimizer and an initial stepsize for

# the gradient descent. Continuing our example from before, below we define a simple loss function

# that is a sum of expectation values returned by ``.op_expval()`` .

# 

def loss_fn(params, circuit, ops, n_samples, key):
    expvals = circuit.op_expval(params, ops, n_samples, key)[0]
    return jnp.sum(expvals)

optimizer = "Adam"
stepsize = 0.001

trainer = iqp.Trainer(optimizer, loss_fn, stepsize)

######################################################################
# Any differentiable loss function expressible in JAX can be defined, but must have a first argument

# ``params`` that corresponds to the optimization parameters of the circuit. To minimize the loss

# function, we call the method ``.train()`` , which requires the number of iterations ``n_iters`` and

# the initial arguments of the loss function ``loss_kwargs`` given as a dictionary object. This

# dictionary must contain a key ``params`` whose corresponding value specifies the initial parameters.

# 

import matplotlib.pyplot as plt

n_iters = 1000
params_init = np.random.normal(0, 1, len(circuit.gates))
n_samples = 100

loss_kwargs = {
    "params": params_init,
    "circuit": circuit,
    "ops": ops,
    "n_samples": n_samples,
    "key": key,
}

trainer.train(n_iters, loss_kwargs)

trained_params = trainer.final_params
plt.plot(trainer.losses) # plot the loss curve
plt.show()

######################################################################
# Generative machine learning tools

# ---------------------------------

# 

# The package contains a module ``gen_qml`` with functionality to train and evaluate generative models

# expressed as ``IqpSimulator`` circuits.

# 

######################################################################
# Training via the maximum mean discrepancy loss

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 

# The Maximum Mean Discrepancy (MMD) [9] is an integral probability metric that measures the

# similarity between two probability distributions, and can serve as a loss function to train

# generative models. It has one parameter called ``sigma``, the bandwidth of the used gaussian kernel.

# 

# This metric is usually calculated with samples from both probability distributions and we have an

# implementation ready for this case in the ``gen_qml`` module.

# 

import iqpopt.gen_qml as gen
from iqpopt.gen_qml.utils import median_heuristic

n_bits = 100
# toy datasets of low weight bitstrings
X1 = np.random.binomial(1, 0.05, size=(100, n_bits))
X2 = np.random.binomial(1, 0.07, size=(100, n_bits))

sigma = median_heuristic(X1) # bandwidth for MMD

mmd = gen.mmd_loss_samples(X1, X2, sigma)
print(mmd)

######################################################################
# This metric can be determined only with expectation values of PauliZ operators, so, when instead of

# samples, we have an ``IqpSimulator`` object, we can also calculate it comparing the circuit's

# distribution with the ``ground_truth``.

# 

# The implementation in the ``gen_qml`` module only needs an additional parameter ``n_ops`` that

# controls the accuracy of this value. For each of these ``n_ops``, an expectation value will be

# calculated with ``n_samples``. The higher the number of operators the better.

# 

# Let's see it in an example with 20 qubits:

# 

n_qubits = 20

# toy dataset of low weight bitstrings (from ground truth p)
ground_truth = np.random.binomial(1, 0.2, size=(100, n_qubits))

gates = local_gates(n_qubits, 2)
circuit = iqp.IqpSimulator(n_qubits, gates)

params = np.random.normal(0, 1/np.sqrt(n_qubits), len(gates))
sigma = median_heuristic(ground_truth) # bandwidth for MMD

mmd = gen.mmd_loss_iqp(params,
                       circuit,
                       ground_truth,
                       sigma,
                       n_ops=1000,
                       n_samples=1000,
                       key=jax.random.PRNGKey(42))
print(mmd)

######################################################################
# Now, similar to what we did a few sections back in *Optimizing a circuit*, this function can be used

# with a ``Trainer`` object to train a quantum generative model given as a parameterized IQP circuit.

# 

# Using the last defined 20 qubits circuit:

# 

X_train = np.random.binomial(1, 0.2, size=(1000, n_qubits))
params_init = np.random.normal(0, 1/np.sqrt(n_qubits), len(gates))
loss = gen.mmd_loss_iqp # MMD loss

loss_kwargs = {
    "params": params_init,
    "iqp_circuit": circuit,
    "ground_truth": X_train, # samples from ground truth distribution
    "sigma": sigma,
    "n_ops": 1000,
    "n_samples": 1000,
    "key": jax.random.PRNGKey(42),
}

trainer = iqp.Trainer("Adam", loss, stepsize=0.01)
trainer.train(n_iters=250, loss_kwargs=loss_kwargs)

trained_params = trainer.final_params
plt.plot(trainer.losses)
plt.show()

######################################################################
# Kernel Generalized Empirical Likelihood

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 

# In order to probe how well a generative model captures a distribution of high-dimensional data, we

# can use the Kernel Generalized Empirical Likelihood [10]. This metric is a measure of the closeness

# of two distributions while also serving as a tool to evaluate mode dropping.

# 

# As before, if samples are available from the generative model, we have a straightforward

# implementation of the KGEL test:

# 

n_qubits = 10

# toy dataset of low weight bitstrings
X = np.random.binomial(1, 0.2, size=(1000, n_qubits))

n_witness = 10
T = X[-n_witness:] # witness points (sampled from the ground truth distribution)
X = X[:-n_witness] # ground truth for the kgel test

# model samples
Y = np.random.binomial(1, 0.3, size=(1000, n_qubits))

sigma = median_heuristic(X) #bandwidth for KGEL
kgel, p_kgel = gen.kgel_opt_samples(T, X, Y, sigma)
print(kgel)

######################################################################
# To evaluate the KGEL for a model given by a ``IqpSimulator`` object, we need to use another

# implementation based on the calculation of expectation values of PauliZ operators. Let's see how

# well the last trained circuit matches the ground truth with the KGEL test:

# 

n_witness = 10
T = X_train[-n_witness:] # witness points (sampled from the ground truth distribution)
X = X_train[:-n_witness] # ground truth for the kgel test

sigma = median_heuristic(X)
kgel, p_kgel = gen.kgel_opt_iqp(circuit,
                                trained_params,
                                T,
                                X,
                                sigma,
                                n_ops=1000,
                                n_samples=1000,
                                key=jax.random.PRNGKey(42))
print(kgel)

######################################################################
# References:

# 

# [1] Michael J Bremner, Richard Jozsa, and Dan J Shepherd. "Classical simulation of commuting

# quantum computations implies collapse of the polynomial hierarchy". In: Proceedings of the Royal

# Society A: Mathematical, Physical and Engineering Sciences 467.2126 (2011), pp. 459-472 (page 1).

# 

# [2] Michael J Bremner, Ashley Montanaro, and Dan J Shepherd. "Average-case complexity versus

# approximate simulation of commuting quantum computations". In: Physical review letters 117.8 (2016),

# p. 080501 (page 1).

# 

# [3] Simon C Marshall, Scott Aaronson, and Vedran Dunjko. "Improved separation between quantum and

# classical computers for sampling and functional tasks". In: arXiv preprint arXiv:2410.20935 (2024)

# (page 1).

# 

# [4] Michael J Bremner, Ashley Montanaro, and Dan J Shepherd. "Achieving quantum supremacy with

# sparse and noisy commuting quantum computations". In: Quantum 1 (2017), p. 8 (page 1).

# 

# [5] Tomoyuki Morimae and Suguru Tamaki. "Additive-error fine-grained quantum supremacy". In: Quantum

# 4 (2020), p. 329 (page 1).

# 

# [6] Louis Paletta, Anthony Leverrier, Alain Sarlette, Mazyar Mirrahimi, and Christophe Vuillot.

# "Robust sparse IQP sampling in constant depth". In: Quantum 8 (2024), p. 1337 (page 1).

# 

# [7] Dolev Bluvstein, Simon J Evered, Alexandra A Geim, Sophie H Li, Hengyun Zhou, Tom Manovitz,

# Sepehr Ebadi, Madelyn Cain, Marcin Kalinowski, Dominik Hangleiter, et al. "Logical quantum processor

# based on reconfigurable atom arrays". In: Nature 626.7997 (2024), pp. 58-65 (pages 1, 2).

# 

# [8] Dominik Hangleiter, Marcin Kalinowski, Dolev Bluvstein, Madelyn Cain, Nishad Maskara, Xun Gao,

# Aleksander Kubica, Mikhail D Lukin, and Michael J Gullans. "Fault-tolerant compiling of classically

# hard IQP circuits on hypercubes". In: arXiv preprint arXiv:2404.19005 (2024) (page 1).

# 

# [9] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola.

# "A Kernel Two-Sample Test". In: Journal of Machine Learning Research 13.25 (2012), pp. 723-773. url:

# http://jmlr.org/papers/v13/gretton12a.html (page 14).

# 

# [10] Suman Ravuri, Mélanie Rey, Shakir Mohamed, and Marc Deisenroth. Understanding Deep Generative

# Models with Generalized Empirical Likelihoods. 2023. arXiv: 2306.09780 [cs.LG]. url:

# https://arxiv.org/abs/2306.09780 (pages 3, 16).

# 

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/erik_recio.txt
