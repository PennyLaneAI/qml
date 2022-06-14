"""
Arithmetic with the Quantum Fourier Transform
==============================================

.. meta::
    :property="og:description": Learn how quantum fourier transform (QFT) can help with basic arithmetic
    :property="og:image": https://pennylane.ai/qml/_images/learning_from_exp_thumbnail.png

*Author: Guillermo Alonso-Linaje. Posted: 14 June 2022*


Throughout all our material we have seen concepts as varied as the
application of quantum computing to the field of machine learning,
quantum chemistry or optimization problems. All of them are problems of
great complexity but, can we apply quantum computing to more fundamental
tasks? Throughout this tutorial we will answer this question by showing
how we can work with basic arithmetic using an important tool such as
the quantum Fourier Transform (QFT).

Basic operations
-----------------

In this case, we will not focus on understanding how the QFT is built,
as we can find a great explanation in the
`Codebook <https://codebook.xanadu.ai/F.1>`__, but we will develop the
intuition of how it works and what applications we can give it.

Arithmetic is the part of mathematics that studies numbers and the
operations that are done with them, that is, we will learn to add,
subtract and multiply numbers. But here something that we must take into
account is that since we are working with qubits (which can take the
values 0 or 1), we will represent the numbers in binary. For the
purposes of this tutorial, we will assume that we are working with
integers and therefore if we have :math:`n` qubits, we will be able to
represent the numbers from :math:`0` to :math:`2^n-1`.

Having said that, the first step is to remember what is the Pennylane
standard for encoding numbers in binary. A binary number can be
represented as a string of 1s and 0s which we will represent as follows:

.. math:: \vert q_0q_1...q_{n-1}\rangle

where :math:`q_0` refers to the most representative bit, so the formula
to obtain the equivalent decimal number will be:

.. math:: m:= \sum_{i = 0}^{n-1}2^{n-1-i}q_i

That means that the number
:math:`\vert 110 \rangle = 1 \times 2^2 + 1\times 2^1+0\times 2^0 = 6`.

Let’s see how we would represent all the numbers with 3 qubits.

"""
##############################################################################
# .. figure:: ../demonstrations/qft_arithmetic/comp_basis.gif
#    :align: center
#    :width: 50%

"""
Note that if the result of an operation is greater than the maximum
value :math:`2^n-1`, we will start again from zero, that is to say, we
will calculate the modulus of the sum. For example, if we have three
qubits and we want to add :math:`6+3`, we will see that we do not have
enough space since :math:`6+3 = 9 > 2^3-1`. The result we will get will
be :math:`9 \pmod 8 = 1`, or :math:`001` in binary. So be sure to use
enough qubits to represent your solutions!

We can use
the\ ```qml.BasisEmbedding`` <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.BasisEmbedding.html>`__
template to obtain the binary representation in a simple way:

"""

import pennylane as qml

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
@qml.compile()
def circuit():
    qml.BasisEmbedding(6, wires=range(3))
    return qml.state()


qml.draw_mpl(circuit, show_all_wires=True)()


######################################################################
# As we can see the qubit 0 is place on top and the rest of the qubit
# below it. With this in mind, what we will do now, with the objective of
# learning about basic arithmetic, is to create an operator
# :math:`\text{Sum}(k)` such that it takes a state :math:`\vert m \rangle`
# and returns the value :math:`\vert m + k\rangle`. We will see that this
# task is especially easy, if instead of working in the computational
# basis, we work in the Fourier basis. Let us quickly recall the
# representation in this basis.
#
# In this basis, all the basic states will be represented with qubits in
# the XY plane of the Bloch sphere each of them rotated by a certain
# amount. How can we know what angle each qubit is rotated to represent a
# certain number? Well, actually is very easy! Suppose we are working with
# :math:`n` qubits and we want to represent the number :math:`m` in
# fourier basis. Then the j-th qubit will have the following phase:
#
# .. math:: \alpha_j = \frac{m\pi}{2^{j}}
#
# Let’s see how to represent the numbers in the case of 3 qubits:
#
##############################################################################
# .. figure:: ../demonstrations/qft_arithmetic/qft_basis.gif
#    :align: center
#    :width: 50%
#
# As we can see, the least significant qubit will advance
# :math:`\frac{1}{8}` of a turn as we increase the number. The next qubit,
# :math:`\frac{1}{4}` turn and finally the most significant qubit will go
# half a turn each time we advance one unit.
#
# The fact that the encoding of numbers is now in phase gives us great
# flexibility in carrying out our arithmetic operations. To see this,
# let’s look at the following situation. We have a number :math:`m`
# encoded in binary and we want to add :math:`k` units, the procedure will
# be as follows:
#
# -  We convert the encoding into Fourier basis by applying QFT on the
#    :math:`\vert m \rangle` state.
# -  We do a phase rotation to each j qubit of :math:`\frac{k}{2^{j}}`
#    with a :math:`R_Z` gate.
# -  We have therefore that the new phases are
#    :math:`\frac{(m + k)\pi}{2^{j}}`.
# -  We apply :math:`\text{QFT}^{-1}` to return to the computational basis
#    and obtain :math:`m+k`.
#

import pennylane as qml
from pennylane import numpy as np

n_wires = 4
dev = qml.device("default.qubit", wires=n_wires, shots=1)


def add_k_fourier(k, wires):
    for j in range(len(wires)):
        qml.RZ(k * np.pi / (2**j), wires=wires[j])


@qml.qnode(dev)
def circuit(m, k):

    qml.BasisEmbedding(m, wires=range(n_wires))  # step 1

    qml.QFT(wires=range(n_wires))  # step 2

    add_k_fourier(k, range(n_wires))  # step 3

    qml.adjoint(qml.QFT)(wires=range(n_wires))  # step 4

    return qml.sample()


circuit(3, 4)


######################################################################
# It is important to point out that it is not necessary to know how the
# QFT is constructed in order to use it. By knowing the properties of the
# new base we can use it in a simple way.
#
# In this particular algorithm, we have had to introduce :math:`k` in a
# classical way. But let us imagine that what we are interested in is that
# another register of qubits determine what is the quantity to be summed,
# i.e., we look for a new operator :math:`\text{Sum}` such that:
#
# .. math:: \text{Sum}\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m+k \rangle
#
# In this case, we can understand the third register (which is initially
# at 0) as a counter, and we will add as many units as :math:`m` and
# :math:`k` indicate. In this case, having the binary decomposition will
# make it simple. If we have :math:`m = \vert q_0q_1q_2 \rangle` we will
# have to add 1 to the counter if :math:`q_2 = 1` and not add anything
# otherwise. Generically we should add :math:`2^{n-i-1}` units if the i-th
# qubit is a 1 and 0 otherwise. As we can appreciate, this idea it is the
# same of the controlled gate concept and we will apply a corresponding
# phase if indeed the control qubit takes state 1.
#

import matplotlib.pyplot as plt

wires_m = [0, 1, 2]
wires_k = [3, 4, 5]
wires_sol = [6, 7, 8, 9]

dev = qml.device("default.qubit", wires=wires_m + wires_k + wires_sol, shots=1)

n_wires = len(dev.wires)


def addition(wires_m, wires_k, wires_sol):

    # prepare sol-qubits to counting
    qml.QFT(wires=wires_sol)

    # add m to the counter
    for i in range(len(wires_m)):
        qml.ctrl(add_k_fourier, control=wires_m[i])(2 ** (len(wires_m) - i - 1), wires_sol)

    # add k to the counter
    for i in range(len(wires_k)):
        qml.ctrl(add_k_fourier, control=wires_k[i])(2 ** (len(wires_k) - i - 1), wires_sol)

    # return to computational basis
    qml.adjoint(qml.QFT)(wires=wires_sol)


@qml.qnode(dev)
def circuit(m, k, wires_m, wires_k, wires_sol):

    # m and k codification
    qml.BasisEmbedding(m, wires=wires_m)
    qml.BasisEmbedding(k, wires=wires_k)

    # apply the addition circuit
    addition(wires_m, wires_k, wires_sol)

    return qml.sample(wires=wires_sol)


print(circuit(7, 3, wires_m, wires_k, wires_sol))

qml.draw_mpl(circuit, show_all_wires=True)(7, 3, wires_m, wires_k, wires_sol)
plt.show()


######################################################################
# Great! We have just seen how to add a number to a counter and in this
# example, when we added :math:`3 + 7` to get :math:`10`, which in binary
# is :math:`1010`. Following the same idea we will see how easily we can
# implement the multiplication. Let’s imagine that we want to multiply
# :math:`m` and :math:`k` and store the result in another register as we
# have done before, that is, we look for the operator Mul such that:
#
# .. math:: \text{Mul}\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m\times k \rangle
#
# To understand the multiplication process, let’s suppose that we have to
# multiply :math:`k:=\sum_{i=0}^{n-1}2^{n-i-1}k_i` and
# :math:`m:=\sum_{j=0}^{l-1}2^{l-j-1}m_i`. In this case, the result would
# be:
#
# .. math:: k \times m = \sum_{i=0}^{n-1}\sum_{j = 0}^{l-1}m_ik_i (2^{n-i-1} \times 2^{l-j-1}),
#
# or in other words, if :math:`k_i = 1` and :math:`m_i = 1` add
# :math:`2^{n-i-1} \times 2^{l-j-1}` units to the counter, where n and l
# are the number of qubits with which we encode m and k respectively.
#

wires_m = [0, 1, 2]
wires_k = [3, 4, 5]
wires_sol = [6, 7, 8, 9, 10]

dev = qml.device("default.qubit", wires=wires_m + wires_k + wires_sol, shots=1)

n_wires = len(dev.wires)


def multiplication(wires_m, wires_k, wires_sol):

    # prepare sol-qubits to counting
    qml.QFT(wires=wires_sol)

    # add m to the counter
    for i in range(len(wires_k)):
        for j in range(len(wires_m)):
            coeff = 2 ** (len(wires_m) + len(wires_k) - i - j - 2)
            qml.ctrl(add_k_fourier, control=[wires_k[i], wires_m[j]])(coeff, wires_sol)

    # return to computational basis
    qml.adjoint(qml.QFT)(wires=wires_sol)


@qml.qnode(dev)
def circuit(m, k):

    # m and k codification
    qml.BasisEmbedding(m, wires=wires_m)
    qml.BasisEmbedding(k, wires=wires_k)

    # Apply multiplication
    multiplication(wires_m, wires_k, wires_sol)

    return qml.sample(wires=wires_sol)


print(circuit(7, 3))

qml.draw_mpl(circuit, show_all_wires=True)(7, 3)
plt.show()


######################################################################
# Awesome! We have multiplied :math:`7 \times 3` and as a result we have
# :math:`10101` that is, :math:`21` in binary.
#


######################################################################
# With this we have already gained a large repertoire of interesting
# operations that we can do but let’s give the idea one more twist and
# apply what we have learned in an example.
#
# Let’s imagine now that what we want is just the opposite, to factor the
# number 21 as a product of two terms, is this something we could do
# following this previous reasoning? The answer is yes! We can make use of
# Grover’s algorithm to amplify the states whose product is the number we
# are looking for. All we would need is to construct the oracle U, i.e. an
# operator such that:
#
# .. math:: U\vert m \rangle \vert k \rangle = \vert m \rangle \vert k \rangle \text{ if }m\times k \not = 21
#
# .. math:: U\vert m \rangle \vert k \rangle = -\vert m \rangle \vert k \rangle \text{ if }m\times k  = 21
#
# The idea of the oracle is as simple as this:
#
# -  use auxiliary registers to store the product
# -  check if the product state is 10101 and in that case change the sign
# -  execute the inverse of the circuit to clear the auxiliary qubits
#

import matplotlib.pyplot as plt

n = 7

wires_m = [0, 1, 2]
wires_k = [3, 4, 5]
wires_sol = [6, 7, 8, 9, 10]

dev = qml.device("default.qubit", wires=wires_m + wires_k + wires_sol)

n_wires = len(dev.wires)


@qml.qnode(dev)
def factorization(n, wires_m, wires_k, wires_sol):

    # Superposition of the input
    for wire in wires_m:
        qml.Hadamard(wires=wire)

    for wire in wires_k:
        qml.Hadamard(wires=wire)

    # Apply the multiplication
    multiplication(wires_m, wires_k, wires_sol)

    # Change sign to n
    qml.BasisEmbedding(2 ** len(wires_sol) - n - 1, wires=wires_sol)
    qml.ctrl(qml.PauliZ, control=wires_sol[:-1])(wires=wires_sol[-1])
    qml.BasisEmbedding(2 ** len(wires_sol) - n - 1, wires=wires_sol)

    # Uncompute multiplication
    qml.adjoint(multiplication)(wires_m, wires_k, wires_sol)

    # Apply Grover Operator
    qml.GroverOperator(wires=wires_m + wires_k)

    return qml.probs(wires=wires_m)


qml.draw_mpl(factorization)(n, wires_m, wires_k, wires_sol)
plt.show()


######################################################################
# A very cool circuit, let’s calculate the probabilities to see each basic
# state!
#

plt.bar(range(2 ** len(wires_m)), factorization(n, wires_m, wires_k, wires_sol))
plt.show()


######################################################################
# By plotting the probabilities of obtaining each basic state we get the
# prime factors are just amplified! Factorization via Grover’s algorithm
# does not achieve exponential improvement as Shor’s algorithm does but we
# can see that the construction is simple and is a great example to
# illustrate basic arithmetic! This will help us in the future to build
# more complicated operators, but until then, let’s keep on learning :)
#
