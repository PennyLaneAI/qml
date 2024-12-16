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
# Creating an IQP circuit with pennylane

# --------------------------------------

# 

######################################################################
# These are circuits comprised of gates :math:`\text{exp}(i\theta_j X_j)`, where the generator

# :math:`X_j` is a tensor product of Pauli X operators acting on some subset of qubits and

# :math:`\theta_j` is a trainable parameter. Input states and measurements are diagonal in the

# computational (Z) basis.

# 

# To define such a circuit, we need a way to represent the parameterized gates. A set of gates will be

# specified by a list

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

# that contains the generators of the gates (each generator with an independent trainable parameter).

# Since all gates have Pauli X type generators, a generator will be represented by a list that

# specifies which qubits are acted on by an X operator. For example,

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

# Here is how to prepare a three qubit circuit with all the possible single and two qubit gates

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

# In order to be more general and have more than one generator with the same trainable parameter, we

# are going to group the generators with a tuple. Therefore, if the first and second generators share

# the same parameter, then

# 

# .. raw:: html

# 

#    <center>

# 

# gates = [([0], [1]), ([2],), ([0,1],), ([0,2],), ([1,2],)]

# 

# .. raw:: html

# 

#    </center>

# 

# As we can see, the rest of the generators are alone in their own tuple. This would be the final form

# of our gates parameter, which will define our circuit from now on.

# 

# Let's now put it in practice and build a pennylane circuit out of this ``gates`` parameter. The

# circuit will be build thanks to the following identity:

# 

# .. math:: \text{exp}(i\theta_j X_j) = H^{\otimes n} \text{exp}(i\theta_j Z_j) H^{\otimes n}

# 

# where :math:`H` is the Hadamard matrix and :math:`Z_j` is the same tensor product of Paulis, but now

# instead of Pauli X there are Pauli Z with the corresponding identities.

# 

# Our pennyalne circuit (with input state :math:`\vert 0 \rangle`) would therefore be the following.

# 

import pennylane as qml
import numpy as np

def penn_iqp_gates(params: np.ndarray, gates: list, n_qubits: int):
    """IQP circuit in pennylane form.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        gates (list): The gates representation of the IQP circuit.
        n_qubits (int): The total number of qubits in the circuit.
    """

    for i in range(n_qubits):
        qml.Hadamard(i)

    for par, gate in zip(params, gates):
        for gen in gate:
            qml.MultiRZ(2*par, wires=gen)

    for i in range(n_qubits):
        qml.Hadamard(i)

######################################################################
# Now, we want to measure our circuit. As we said, all measurements are diagonal in the computational

# (Z) basis, and we will therefore represent them with a bitstring. 1s, wherever there are Pauli Z,

# and 0s, wherever there are identities. In this way, an operator such as:

# 

# .. math:: O = Z \otimes I \otimes Z

# 

# will be converted to

# 

# .. raw:: html

# 

#    <center>

# 

# op = [1,0,1]

# 

# .. raw:: html

# 

#    </center>

# 

# The final circuit that allow us to measure these type of expectation values is the following.

# 

def penn_obs(op: np.ndarray) -> qml.operation.Observable:
    """Returns a pennylane observable from a bitstring representation.

    Args:
        op (np.ndarray): Bitstring representation of the Z operator.

    Returns:
        qml.Observable: Pennylane observable.
    """
    for i, z in enumerate(op):
        if i==0:
            if z:
                obs = qml.Z(i)
            else:
                obs = qml.I(i)
        else:
            if z:
                obs @= qml.Z(i)
    return obs

def penn_iqp_circuit(params: np.ndarray, gates: list, op: np.ndarray, n_qubits: int) -> qml.measurements.ExpectationMP:
    """Defines the circuit that calculates the expectation value of the operator with the IQP circuit with pennylane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        gates (list): The gates representation of the IQP circuit.
        op (np.ndarray): Bitstring representation of the Z operator.
        n_qubits (int): The total number of qubits in the circuit.

    Returns:
        qml.measurements.ExpectationMP: Pennylane circuit with an expectation value.
    """
    penn_iqp_gates(params, gates, n_qubits)
    obs = penn_obs(op)
    return qml.expval(obs)

def penn_iqp_op_expval(params: np.ndarray, gates: list, op: np.ndarray, n_qubits: int) -> float:
    """Calculates the expectation value of the operator with the IQP circuit with pennylane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        gates (list): The gates representation of the IQP circuit.
        op (np.ndarray): Bitstring representation of the Z operator.
        n_qubits (int): The total number of qubits in the circuit.

    Returns:
        float: Expectation value.
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)
    penn_iqp_circuit_exe = qml.QNode(penn_iqp_circuit, dev)
    return penn_iqp_circuit_exe(params, gates, op, n_qubits)

######################################################################
# With this, we can now calculate all expectation of Pauli Z tensors of any IQP circuit we can think

# of. Let's see an example.

# 

n_qubits = 3
gates = [([0], [1]), ([2],), ([0,1],), ([0,2],), ([1,2],)] # same gates as in the example above
op = np.array([1,0,1]) # operator ZIZ
params = np.random.rand(len(gates)) # random parameters for all the gates (remember we have 5 gates with 6 generators in total)

penn_op_expval = penn_iqp_op_expval(params, gates, op, n_qubits)
print(penn_op_expval)

######################################################################
# Estimating expectation values with IQPopt

# -----------------------------------------

# 

# With this new package we can perform the same calculations our pennylane circuit above is able to

# do. Although, this time, only with approximations instead of exact values. The gain is that we can

# work with a very large amount of qubits.

# 

# After some calculations, one can arrive at the following expression when calculating the expectation

# value of any tensor product of Pauli Z operators

# 

# .. math:: \langle Z_{\boldsymbol{a}} \rangle = \mathbb{E}_{\boldsymbol{z}\sim U}\Big[ \cos\Big(\sum_j \theta_{j}(-1)^{\boldsymbol{g}_{j}\cdot \boldsymbol{z}}(1-(-1)^{\boldsymbol{g}_j\cdot \boldsymbol{a}}\Big) \Big],

# 

# where :math:`\boldsymbol{a}` is the bitstring form of the operator we want to calculate the

# expectation value of, :math:`\boldsymbol{z}` are bitstring samples from the multi-qubit system taken

# from the uniform distribution, :math:`\theta_{j}` are the trainable parameters and

# :math:`\boldsymbol{g}_{j}` are the different generators also in bitstring form. The dots are dot

# products between the bitstrings taken as vectors of 1s and 0s.

# 

# We could classically calculate this expression exactly, but it would require an exponentially large

# amount of :math:`\boldsymbol{z}` samples with the increasing number of qubits. To aliviate this, we

# can replace the expectation with an empirical mean and compute an unbiased estimate of

# :math:`\langle Z_{\boldsymbol{a}} \rangle` efficiently. That is, if we sample a batch of :math:`s`

# bit strings :math:`\{\boldsymbol{z}_i\}` from the uniform distribution and compute the sample mean

# 

# .. math:: \hat{\langle Z_{\boldsymbol{a}}\rangle} = \frac{1}{s}\sum_{i}\cos\Big(\sum_j \theta_j(-1)^{\boldsymbol{g}_j\cdot \boldsymbol{z}_i}(1-(-1)^{\boldsymbol{g}_j\cdot \boldsymbol{a}})\Big),

# 

# we obtain an unbiased estimate :math:`\hat{\langle Z_{\boldsymbol{a}}\rangle}` of

# :math:`\langle Z_{\boldsymbol{a}}\rangle`; i.e. we have that

# :math:`\mathbb{E}[\hat{\langle Z_{\boldsymbol{a}}\rangle}] = \langle Z_{\boldsymbol{a}}\rangle`. The

# error of this approximation is well known since, by the central limit theorem, the standard

# deviation of the sample mean of a bounded random variable decreases as

# :math:`\mathcal{O}(1/\sqrt{s})`.

# 

# Let's see now how to use the IQPopt package to calculate expectation values, using the same

# arguments we used in the previous example. First, we create the circuit object with ``IqpSimulator``

# only with the number of qubits of the circuit and the ``gates`` parameter already explained:

# 

import iqpopt as iqp

small_circuit = iqp.IqpSimulator(n_qubits, gates)

######################################################################
# Now, we use the class method ``IqpSimulator.op_expval()``, which is the one that provides these

# estimates. This function requires a parameter array ``params``, a PauliZ operator specified by its

# binary representation ``op``, a new parameter ``n_samples`` (the number of samples :math:`s`) that

# controls the precision of the approximation (the more the better), and a JAX pseudo random number

# generator key to seed the randomness of the sampling. It returns the expectation value estimate as

# well as its standard error.

# 

# Using the same ``params`` and ``op`` as before:

# 

import jax

n_samples = 1000
key = jax.random.PRNGKey(42)

expval, std = small_circuit.op_expval(params, op, n_samples, key)

print(expval, std)

######################################################################
# Since the calculation on ``iqpopt``\ 's side is stochastic, the result is not exactly the same than

# the one obtained with pennylane. But, as we can see, they are within std error. You can try

# increasing ``n_samples`` in order to obtain closer approximations.

# 

# This function also allows for fast batch evaluation of expectation values. If we specify a batch of

# operators ``ops`` by an array, we can batch evaluate the expectation values and errors in parallel

# with the same syntax.

# 

ops = np.array([[1,0,0],[0,1,0],[0,0,1]]) # batch of single qubit Pauli Zs

expvals, stds = small_circuit.op_expval(params, ops, n_samples, key)

print(expvals, stds)

######################################################################
# With pennylane it would be very time consuming to pass the 30 qubit mark, but with IQPopt, we can

# easily go way further than that.

# 

from iqpopt.utils import local_gates

n_qubits = 1000
gates = local_gates(n_qubits, 1) # 1000 single qubit generators with independent trainable parameters

large_circuit = iqp.IqpSimulator(n_qubits, gates)

params = np.random.rand(len(gates))
op = np.random.randint(0, 2, n_qubits)
n_samples = 1000
key = jax.random.PRNGKey(42)

expval, std = large_circuit.op_expval(params, op, n_samples, key)

print(expval, std)

######################################################################
# Sampling and probabilities

# --------------------------

# 

######################################################################
# We can also view a parameterized IQP circuit as a generative model that generates samples of binary

# vectors according to the distribution

# 

# .. math:: q_{\boldsymbol{\theta}}(\boldsymbol{x}) \equiv q(\boldsymbol{x}\vert\boldsymbol{\theta})=\vert\bra{\boldsymbol{x}}U(\boldsymbol{\theta})\ket{0}\vert^2.

# 

# For a low amount of qubits, we can use Pennylane's arsenal and know the output probabilities of the

# circuit as well as sample from it. Note that there is not an efficient algorithm to do these, so,

# for large numbers of qubits, these methods will either take too long or not work at all.

# 

# These functions are already implemented in the ``IqpSimulator`` object. The ``.probs()`` method

# works as it does in pennylane, the returned array of probabilities is in lexicographic order.

# 

sample = small_circuit.sample(params, shots=1)
print(sample)

probabilities = small_circuit.probs(params)
print(probabilities)

sample = large_circuit.sample(params, shots=1)
print(sample)

probabilities = large_circuit.probs(params)
print(probabilities)

######################################################################
# As we can see, we can't sample or know the probabilities of the circuit for the large one. The only

# efficient approximation algorithm we have is for the calculation of expectation values. In the

# following figure, you can see how the time scales with the different methods when we increase the

# number of qubits. The calculation of expectation values is the only one that remains linear instead

# of increasing exponentially.

# 

######################################################################
# .. figure:: ../_static/demonstration_assets/iqpopt/sample_time.png

#    :alt: Time vs n_qubits

# 

#    Time vs n_qubits

######################################################################
# Optimizing an IQPopt circuit

# ----------------------------

# 

######################################################################
# Circuits can be optimized via a separate ``Trainer`` class. To instantiate a trainer object we first

# define a loss function (also called an objective function), an optimizer and an initial stepsize for

# the gradient descent. Continuing our ``small_circuit`` example from before, below we define a simple

# loss function that is a sum of expectation values returned by ``.op_expval()`` .

# 

import jax.numpy as jnp

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
params_init = np.random.normal(0, 1, len(small_circuit.gates))
n_samples = 100

loss_kwargs = {
    "params": params_init,
    "circuit": small_circuit,
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
# References:

# 

# [1] Michael J Bremner, Richard Jozsa, and Dan J Shepherd. "Classical simulation of com- muting

# quantum computations implies collapse of the polynomial hierarchy". In: Proceed- ings of the Royal

# Society A: Mathematical, Physical and Engineering Sciences 467.2126 (2011), pp. 459-472 (page 1).

# 

# [2] Michael J Bremner, Ashley Montanaro, and Dan J Shepherd. "Average-case complex- ity versus

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

# [9] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexan- der Smola.

# "A Kernel Two-Sample Test". In: Journal of Machine Learning Research 13.25 (2012), pp. 723-773. url:

# http://jmlr.org/papers/v13/gretton12a.html (page 14).

# 

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/erik_recio.txt
