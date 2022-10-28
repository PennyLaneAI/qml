r"""
Perturbative Gadgets for Variational Quantum Algorithms  
==========================================

*Author: Simon Cichy*

In this tutorial, we will explore the application of perturbative gadgets in 
variational quantum algorithms to outgo the issue of cost function dependent
barren plateaus, as proposed in Ref. [#cichy2022]_ 

Some context
------------

Barren plateaus refer to the phenomenon where the gradients of the cost function
decay exponentially with the size of the problem. Essentially, the cost 
landscape becomes flat, with exception of some small regions, e.g., around
the minimum. That is a problem because increasing the resolution on a quantum 
computer comes at the cost of more measurements, and an exponential number of 
measurement would render the algorithm impractical.
If you are not familiar yet with the concept of barren plateaus, I recommend you
first check out the demonstrations on :doc:`barren plateaus </demos/tutorial_barren_plateaus>`
and :doc:`avoiding barren plateaus with local cost functions </demos/tutorial_local_cost_functions>`.

As presented in the second, barren plateaus are more severe when using global
cost functions compared to local ones. 
We want to explore this topic further and learn about one possible mitigation
strategy.  
Thinking about VQE applications, let us consider cost functions that are 
expectation values of Hamiltonians like

.. math:: C(\theta) = \operatorname{Tr} \left[ H V(\theta) |00\ldots 0\rangle \langle 00\ldots 0| V(\theta)^\dagger\right].

Here :math:`|00\ldots 0\rangle` is our initial state, 
:math:`V(\theta)` is the circuit ansatz and :math:`H` the Hamiltonian
to minimize.  

In some cases, it is easy to find a local cost function to substitute a global
one, which still has the same ground state. 
For instance, one can verify that the local cost function built from 

.. math:: H_L = \mathbb{I} - \frac{1}{n} \sum_j |0\rangle \langle 0|_j 

has the same ground state as the global one

.. math:: H_G = \mathbb{I} - |00\ldots 0\rangle \langle 00\ldots 0| 

and that is 

.. math:: |\psi (\theta_{min}) \rangle =  |00\ldots 0\rangle.

However, it is not always so simple. 
What if we want to find the minimum eigenenergy of 
:math:`H = X \otimes X \otimes Y \otimes Z + Z \otimes Y \otimes X \otimes X` ?  
It is not always trivial to construct a local cost 
function which has the same minimum as some other cost function of interest,
global however. 
That is where perturbative gadgets come into play, and we will see how.

The definitions
---------------
Perturbative gadgets are a common tool in adiabatic quantum computing. 
Their goal is to find a Hamiltonian with local interactions which mimics
some other Hamiltonian with more complex couplings. 
The later is the one they would ideally want to implement for their
computation, that's why it is called the computational Hamiltonian, but can not
since it is hard to implement more than few-body interactions in hardware.
This is done by increasing the dimension of the Hilbert space (i.e. the number 
of qubits) and "encoding" the computational Hamiltonian in the low-energy 
subspace of a so-called gadget Hamiltonian.
Let us now construct such a gadget Hamiltonian taylored for VQE applications.  

First, we start from a target Hamiltonian which is a linear combination of 
Pauli words, acting on :math:`k` qubits

.. math:: H^\text{target} = \sum_i c_i h_i

with :math:`h_i = \sigma_{i,1} \otimes \sigma_{i,2} \otimes \ldots \otimes \sigma_{i,k}`
and :math:`\sigma_{i,j} \in \{ X, Y, Z \}`, :math:`c_i \in \mathbb{R}`.  
Now we construct the gadget Hamiltonian.
For each term :math:`h_i`, we will need :math:`k` additional qubits which we 
call auxiliary qubits, and add two terms to the Hamiltonian: 
an "unperturbed" part :math:`H^\text{aux}_i` and a perturbation :math:`V_i` 
of strength :math:`\lambda`. 
The unperturbed part penalizes each of the newly added qubits for not being in 
the :math:`|0\rangle` state

.. math:: H^\text{aux}_i = \sum_{j=1}^k |1\rangle \langle 1|_{i,j} = \sum_{j=1}^k \frac{1}{2}(\mathbb{I} - Z_{i,j}).

On the other hand, the perturbation part implements one of the operators in the Pauli word
:math:`\sigma_{i,j}` on the corresponding qubit of the computational register and a 
pair of Pauli :math:`X` gates on two of the auxiliary qubits

.. math:: V_i = \sum_{j=1}^k c_{i,j} \sigma_{i,j} \otimes X_{i,j} \otimes X_{i,(j+1) \mathrm{mod }k}.

In the end, 

.. math:: H^\text{gad} = \sum_{i} \left( H^\text{aux}_i + \lambda V_i \right)


To give an idea, this is what would result from working with a Hamiltonian
acting on a total of :math:`8` qubits, that has :math:`3` terms, each of them being a 
:math:`4`-body interaction. 

.. figure:: ../demonstrations/barren_gadgets/gadget-terms-tutorial.png
    :align: center
    :width: 90%

For each of the terms :math:`h_1`, :math:`h_2` and :math:`h_3` we add :math:`4` auxiliary qubits.
In the end, our gadget Hamiltonian thus acts on :math:`8+3\cdot 4 = 20` qubits.
The penalization (red) acts only on the auxiliary registers, penalizing each 
qubit individually, while the perturbations couple target with auxiliary qubits.

As shown in Ref. [#cichy2022]_, this construction results in a spectrum that is similar
to that of the original Hamiltonian for low energies. 
This means that minimizing the gadget Hamiltonian, if reaching its global
minimum, the resulting state will be close to the global minimum of 
:math:`H^\text{target}` too.
Since it is a local cost function, it is better behaved with respect to 
barren plateaus than the global cost, so it is more trainable.
As a result, one can mitigate the onset of cost function dependent barren 
plateaus by substituting the global cost function with the resulting gadget
and using that for training instead. That is what we will do in the following
of this tutorial.
"""

##############################################################################
# First, a few imports. Pennylane and numpy of course, and additionally a few
# functions specific to our tutorial. 
# The ``PerturbativeGadget`` class allows to generate the gadget Hamiltonian
# from a user-given target Hamiltonian in an automated way. 
# For those who want to check the inner workings,
# you can find the code here (
# :download:`barren_gadgets.py <../demonstrations/barren_gadgets/barren_gadgets.py>` 
# ).
# The functions ``get_parameter_shape``, ``generate_random_gate_sequence`` and
# ``build_ansatz`` (for the details:
# :download:`layered_ansatz.py <../demonstrations/barren_gadgets/layered_ansatz.py>` 
# ) are there to build the parapeterized quantum circuit we use in this demo.
# The first computes the shape of the array of trainable parameters that the 
# circuit will need. The second generates a random sequence of Pauli rotations
# from :math:`\{R_X, R_Y, R_Z\}` with the right dimension.
# Finally, ``build_ansatz`` puts the pieces together. 

import pennylane as qml
from pennylane import numpy as np
from barren_gadgets.barren_gadgets import PerturbativeGadgets
from barren_gadgets.layered_ansatz import (
    generate_random_gate_sequence,
    get_parameter_shape,
    build_ansatz,
)

np.random.seed(42)

##############################################################################
# Now, let's take the example given above
#
# .. math::  H = X \otimes X \otimes Y \otimes Z + Z \otimes Y \otimes X \otimes X.
#
# First we construct our target Hamiltonian in Pennylane.
# For this, we use the
# `qml.Hamiltonian <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.Hamiltonian.html>`_
# class

Hcomp = qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliZ(3) \
      + qml.PauliZ(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliX(3)

##############################################################################
# Now we can check that we constructed what we wanted.

print(Hcomp)

##############################################################################
# We indeed have a Hamiltonian composed of two terms, with the expected Pauli
# words.
# Next, we can construct the corresponding gadget Hamiltonian.
# Using the class ``PerturbativeGadgets``, we can automatically
# generate the gadget Hamiltonian from the computational Hamiltonian.
# The object gadgetizer will contain all the information about the settings of
# the gadgetization procedure (there are quite a few knobs one can tweak on,
# but we'll skip that for now).
# Then, the method gadgetize takes a qml.Hamiltonian object and generates the
# corresponding gadget Hamiltonian.

gadgetizer = PerturbativeGadgets()
Hgad = gadgetizer.gadgetize(Hcomp)
print(Hgad)

##############################################################################
# So, let's see what we got.
# We started with 4 computational qubits (labelled 0 to 3) and two 4-body terms.
# Thus we get 4 additional qubits twice (4 to 11).
# The first 16 elements of our Hamiltonian correspond to the unperturbed part.
# The last 8 are the perturbation. They are a little scambled, but one can
# recognize the 8 Paulis from the computational Hamiltonian on the qubits 0 to 3
# and the cyclic pairwise :math:`X` structure on the auxiliaries.
# Indeed, there are :math:`(X_4X_5, X_5X_6, X_6X_7, X_7X_4)` and
# :math:`(X_8X_9, X_9X_{10}, X_{10}X_{11}, X_{11}X_8)`.

##############################################################################
# Training with the gadget Hamiltonian
# -----------------------------------
# Now that we have a little intuition on how the gadget Hamiltonian construction
# works, we will use it to train.
# Classical simulation of qubit systems is expensive, so we will simplify further
# to a computational Hamiltonian with a single term, and show that using the
# gadget Hamiltonian for training allows us to minimize the target Hamiltonian.
# So, let us construct the two Hamiltonians of interest

Hcomp = 1 * qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
perturbation_factor = 10
gadgetizer = PerturbativeGadgets(perturbation_factor)
Hgad = gadgetizer.gadgetize(Hcomp)

##############################################################################
# Then we need to set up our variational quantum algorithm.
# That is we choose a circuit ansatz with randomly initialized weights,
# the training cost function, the optimizer with its step size, and the number of
# optimization steps and the device to run the circuit on.
# As ansatz, we will use a variation of the
# `qml.SimplifiedTwoDesign <https://pennylane.readthedocs.io/en/latest/code/api/pennylane.SimplifiedTwoDesign.html#pennylane.SimplifiedTwoDesign>`_,
# which was proposed in previous
# works on cost function dependent barren plateaus [#cerezo2021]_.
# I will skip the details of the construction, since it is not our focus here,
# and just show how it looks like.
# Here is the circuit for a small example

shapes = get_parameter_shape(n_layers=3, n_wires=5)
init_weights = [np.pi / 4] * shapes[0][0]
weights = np.random.uniform(0, np.pi, size=shapes[1])


@qml.qnode(qml.device("default.qubit", wires=range(5)))
def display_circuit(weights):
    build_ansatz(initial_layer_weights=init_weights, weights=weights, wires=range(5))
    return qml.expval(qml.PauliZ(wires=0))


qml.draw_mpl(display_circuit)(weights)

##############################################################################
# Now we build the circuit for our actual experiment

# Total number of qubits: computational + auxiliary
num_qubits = 4 + 2 * 4

# Other parameters of the ansatz: weights and gate sequence
shapes = get_parameter_shape(n_layers=num_qubits, n_wires=num_qubits)
init_weights = [np.pi / 4] * shapes[0][0]
weights = np.random.uniform(0, np.pi, size=shapes[1])
random_gate_sequence = generate_random_gate_sequence(qml.math.shape(weights))

##############################################################################
# For the classical optimization, we will use the standard gradient descent
# algorithm, and perform 500 iterations. For the quantum part, we will simulate
# our circuit using the
# `default.qubit <https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html>`_
# simulator.

opt = qml.GradientDescentOptimizer(stepsize=0.1)
max_iter = 500
dev = qml.device("default.qubit", wires=range(num_qubits))

##############################################################################
# Finally, we will use two cost functions.
# For each we create a QNode.
# The first, the training cost, is the loss function of the optimization,
# that's the one the gradient descent will actually try to minimize.
# For the training, we use the gadget Hamiltonian.
# Then we also define a monitoring cost, based on the target Hamiltonian.
# We will evaluate it's value at each iteration for monitoring purposes, but it
# will not be used in the optimization.


@qml.qnode(dev)
def training_cost(weights):
    build_ansatz(
        initial_layer_weights=init_weights,
        weights=weights,
        wires=range(num_qubits),
        gate_sequence=random_gate_sequence,
    )
    return qml.expval(Hgad)


@qml.qnode(dev)
def monitoring_cost(weights):
    build_ansatz(
        initial_layer_weights=init_weights,
        weights=weights,
        wires=range(num_qubits),
        gate_sequence=random_gate_sequence,
    )
    return qml.expval(Hcomp)


##############################################################################
# The idea is that if we reach the global minimum for the gadget Hamiltonian, we
# should also be close to the global minimum of the target Hamiltonian, which is
# what we are ultimately interested in.
# To be able to look how it went and maybe plot it, we will save the costs values
# at each iteration.

costs_lists = {}
costs_lists["training"] = [training_cost(weights)]
costs_lists["monitoring"] = [monitoring_cost(weights)]

##############################################################################
# Now everything is set up, let's run the optimization and see how it goes.
# Careful, this will take a while.

print(
    f"Iteration = {0:5d} of {max_iter:5d} | "
    + "Training cost = {:12.8f} | ".format(costs_lists["training"][-1])
    + "Monitoring cost = {:12.8f} | ".format(costs_lists["monitoring"][-1])
)
for it in range(max_iter):
    weights = opt.step(training_cost, weights)
    costs_lists["training"].append(training_cost(weights))
    costs_lists["monitoring"].append(monitoring_cost(weights))
    if (it + 1) % 50 == 0:
        print(
            f"Iteration = {it+1:5d} of {max_iter:5d} | "
            + "Training cost = {:12.8f} | ".format(costs_lists["training"][-1])
            + "Monitoring cost = {:12.8f} | ".format(costs_lists["monitoring"][-1])
        )

import matplotlib.pyplot as plt
plt.style.use("seaborn")

plt.figure()
plt.plot(costs_lists["training"])
plt.plot(costs_lists["monitoring"])
plt.legend(["training", "monitoring"])
plt.xlabel("Number of iterations")
plt.ylabel("Cost values")
plt.show()

##############################################################################
# Since our example of target Hamiltonian is a single Pauli string, we know
# without needing any training that it has only :math:`\pm 1` eigenvalues.
# It is a very simple example but we see that the training of our circuit using
# the gadget Hamiltonian as cost function did indeed allow to reach the
# global minimum of the target cost function.  
# 
# Now that you have an idea of how you can use perturbative gadgets in 
# variational quantum algorithms, you can try applying it to more interesting
# problems! However, be reminded of the exponential scaling of classical 
# simulations of quantum systems, adding linearly many auxiliary qubits
# quickly becomes a lot.
# For those interested in the theory behind it or more formal statements of 
# "how close" the results using the gadget are from the targeted ones, check out
# the original paper [#cichy2022]_ .
# There you will also find a more general recipe to design other gadget 
# constructions with similar properties.  
# Also, the complete repository with explanations on how to reproduce the 
# figures can be found from the paper
# `here <https://github.com/SimonCichy/barren-gadgets>`_ . 

##############################################################################
# References
# ----------
#
# .. [#cichy2022]
#
#    Cichy, S., Faehrmann, P.K., Khatri, S., Eisert, J.
#    "A perturbative gadget for delaying the onset of barren plateaus in variational quantum algorithms." `arXiv:2210.03099
#    <https://arxiv.org/abs/2210.03099>`__, 2022.
#
# .. [#cerezo2021]
#
#    Cerezo, M., Sone, A., Volkoff, T. et al.
#    "Cost function dependent barren plateaus in shallow parametrized quantum circuits." `Nat Commun 12, 1791
#    <https://doi.org/10.1038/s41467-021-21728-w>`__, 2021.
#
# About the author
# ----------------
#
# .. bio:: Simon Cichy
#    :photo: ../_static/authors/simon_cichy.jpg
# 
#    Simon is a graduate in quantum engineering and PhD student at the
#    Freie Universität Berlin. He is interested in quantum machine learning
#    and near-term quantum algorithms in general. 
#    You can find out more about him `here <https://simoncichy.github.io/>`_ .
