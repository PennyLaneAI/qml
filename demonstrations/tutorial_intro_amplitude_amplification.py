r"""Intro to Amplitude Amplification (and its variants)
=======================================================================

Grover's algorithm is one of the most important algorithms developed in quantum computing. This technique belongs to a
much broader category of algorithms called Amplitude Amplification. In this demo, we will make an introduction to the
general problem by seeing how the idea proposed by Grover can be generalized and we will solve some of its limitations
with variants such as fixed-point quantum search.

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/OGthumbnail_large_AmplitudeAmplification_2024-04-29.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Amplitude Amplification
-------------------------

Our goal is to prepare an unknown state :math:`|\phi\rangle` using some known property of that state.
A good first approach is to use a circuit :math:`U` to generate a state :math:`U|0\rangle := |\Psi\rangle`
that "contains" some amount of the target state :math:`|\phi\rangle`. Generically we can represent
:math:`|\Psi\rangle` in the computational basis as:

.. math::
    |\Psi\rangle = \sum_i c_i |i\rangle,

but we can do better. One choice is to make :math:`|\phi\rangle` an element of the basis. We can then write

.. math::
    |\Psi\rangle = \alpha |\phi\rangle + \beta |\phi^{\perp}\rangle,

where  :math:`|\phi^{\perp}\rangle` is some state orthogonal
to :math:`|\phi\rangle` and :math:`\alpha, \beta \in \mathbb{R}`. A great advantage of representing the
state :math:`|\Psi\rangle` in this way is that we can now visualize it
in a two-dimensional space making it easier to manipulate the state. This representation is even simpler than a Bloch
sphere since the amplitudes are real numbers --- we can visualize all operations inside a circle:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp1.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

We are going to *amplify* the amplitude :math:`\alpha` to get closer
to :math:`|\phi\rangle`, hence the name Amplitude Amplification [#ampamp]_. We will try to find an operator that moves
the initial vector :math:`|\Psi\rangle` as close to :math:`|\phi\rangle` as possible.

Finding the operators
~~~~~~~~~~~~~~~~~~~~~

It would be enough if we could create a rotation gate in this subspace, then rotate the initial state counterclockwise
by :math:`\pi/2 -\theta` to obtain :math:`|\phi\rangle`.  However, we don't explicitly know :math:`|\phi\rangle`,
so it's unclear how this could be done.  This is where a great idea is born: What if instead of rotations we think of
reflections?

We could think of reflections with respect to three states: :math:`|\phi\rangle`,  :math:`|\phi^{\perp}\rangle`,
or :math:`|\Psi\rangle`.

The main insight of the Amplitude Amplification algorithm is that there is a sequence of **two reflections** that helps
us in this task.The first is the reflection with respect to :math:`|\phi^{\perp}\rangle` and the second with respect
to :math:`|\Psi\rangle`.

Let's go step by step:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp2.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Reflections with respect to :math:`|\phi^{\perp}\rangle`.

After applying the first reflection we are moving away from :math:`|\phi\rangle`, why do that?
Sometimes it is necessary to take a step backwards in order to take two steps
forward, and that is exactly what we will do.
The :math:`|\phi^{\perp}\rangle` reflection  may seem somewhat complex to create since we do not have access
to such a state. However, the operator is well-defined:

.. math::
     \begin{cases}
     R_{\phi^{\perp}}|\phi\rangle = -|\phi\rangle, \\
     R_{\phi^{\perp}}|\phi^{\perp}\rangle = |\phi^{\perp}\rangle.
     \end{cases}

Amplitude Amplification usually assumes access to an oracle implementing this reflection operator.
For example, in a search problem, this is the usual Grover oracle that "marks" the target state with a phase flip.
In practice, the way to explicitly build the oracle is just a classic check:
if the given state meets the known property, we change its sign. This will become clearer later with an example.


The second reflection is the one with respect to :math:`|\Psi\rangle`. This operator is easier to build since
we know the operator :math:`U` that generate it. This can be built directly in PennyLane with :class:`~.pennylane.Reflection`.

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp3.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Reflections with respect to :math:`|\Psi\rangle`.


These two reflections are equivalent to rotating the state by :math:`2\theta` degrees from its original position,
where :math:`\theta=\arcsin(\alpha)` is the angle that defines the initial state. To amplify the amplitude and
approach the target state, we perform this sequence of rotations multiple times. More precisely, we repeat them:

.. math::
    k = \frac{\pi}{4 \arcsin \alpha}-\frac{1}{2}.

Amplitude Amplification in PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After looking at the theory, let's take a look at a practical example in PennyLane. Let's solve the zero-sum problem.
In this problem we are given a list of :math:`n` integers. Our goal is to find the subsets of numbers
whose sum is :math:`0`. Let us define our values:
"""

n = 6
values = [1, -2, 3, 4, 5, -6]

##############################################################################
# The subset :math:`[1,5,-6]`, is a solution, but finding all of them is an expensive task.
# We will use Amplitude Amplification to solve the problem.
# First we define a binary variable :math:`x_i` that takes the value :math:`1` if we include the i-th element in the
# subset and :math:`0` otherwise.
# We encode the i-th variable in the i-th qubit of a quantum state, so for instance, :math:`|100011\rangle`
# represents the subset above. We can now define the initial state as:
#
# .. math::
#   |\Psi\rangle = \frac{1}{\sqrt{2^n}}\sum_{i=0}^{2^n-1}|i\rangle.
#
# This is equivalent to the combination of all possible subsets and therefore, our searched states are "contained"
# in there. This is a state that we can generate by applying Hadamard gates.

import pennylane as qml
import matplotlib.pyplot as plt
plt.style.use('pennylane.drawer.plot')

@qml.prod
def U(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    U(wires = range(n))
    return qml.state()

output = circuit()[:64].real

plt.bar(range(len(output)), output)
plt.ylim(-0.4, 0.6)
plt.ylabel("Amplitude")
plt.xlabel("|x⟩")
plt.axhline(0, color='black', linewidth=1)
plt.show()



##############################################################################
# Initially, the probability of getting any basis state is the same.
# The next step is to mark those elements that satisfy our property --- that the sum of the subset is :math:`0`.
# To do this we must create the following auxiliary function:
#
# .. math::
#     \text{Sum}|\text{subset}\rangle|0\rangle = |\text{subset}\rangle|\sum v_ix_i\rangle,
#
# where we store in the second register the sum of the subset.
# To see the details of how to build this operation take a look to `Basic arithmetic with the QFT <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics/>`_.
#

import numpy as np

def Sum(wires_subset, wires_sum):

    qml.QFT(wires = wires_sum)
    for i, value in enumerate(values):
        for j in range(len(wires_sum)):
            qml.CRZ(value * np.pi / (2 ** j), wires=[wires_subset[i], wires_sum[j]])
    qml.adjoint(qml.QFT)(wires=wires_sum)

##############################################################################
# With this operator we can mark the searched elements. To do this, we apply :math:`\text{Sum}` , then we change sign
# to those states whose sum has been 0 and then we apply the inverse of the sum to clean the auxiliary qubits
# (otherwise it could produce undesired results when applying Amplitude Amplification).
#

@qml.prod
def oracle(wires_subset, wires_sum):

    Sum(wires_subset, wires_sum)
    qml.FlipSign(0, wires=wires_sum)
    qml.adjoint(Sum)(wires_subset, wires_sum)


@qml.qnode(dev)
def circuit():

    U(wires=range(n))                 # Generate initial state
    oracle(range(n), range(n, n+5))   # Apply oracle

    return qml.state()


output = circuit()[0::2 ** 5].real
plt.bar(range(len(output)), output)
plt.ylim(-0.4, 0.6)
plt.ylabel("Amplitude")
plt.xlabel("|x⟩")
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# Great, we have flipped the sign of the searched states without knowing what they are, simply by making use of their
# property. The next step is to reflect on the :math:`|\Psi\rangle` state, defining the final circuit as:
#
# .. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/sum_zero.jpeg
#    :align: center
#    :width: 60%
#    :target: javascript:void(0)
#
# Let's build it with PennyLane:


@qml.qnode(dev)
def circuit():

    U(wires=range(n))                  # Generate initial state
    oracle(range(n), range(n, n + 5))  # Apply oracle
    qml.Reflection(U(wires=range(n)))    # Reflect on |Psi>

    return qml.state()


output = circuit()[0::2 ** 5].real
plt.bar(range(len(output)), output)
plt.ylim(-0.4, 0.6)
plt.ylabel("Amplitude")
plt.xlabel("|x⟩")
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# The four peaks are obtained in :math:`0`, :math:`27`, :math:`35` and :math:`61`, whose binary
# representation corresponds with :math:`|000000\rangle`, :math:`|011011\rangle`, :math:`|100011\rangle` and :math:`|111101\rangle` respectively.
# These states satisfy the property that the sum of the subset is :math:`0`.
#
# The combination of this two reflections is
# implemented in PennyLane as :class:`~.AmplitudeAmplification`, template that we will use to see the evolution of the
# state as a function of the number of iterations.
@qml.qnode(dev)
def circuit(iters):

    U(wires=range(n))

    qml.AmplitudeAmplification(U = U(wires = range(n)),
                               O = oracle(range(n), range(n, n + 5)),
                               iters = iters)

    return qml.probs(wires = range(n))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i in range(1,9,2):
    output = circuit(iters=i)
    ax = axs[i // 4, i //2 % 2]
    ax.bar(range(len(output)), output)
    ax.set_ylim(0, 0.6)
    ax.set_title(f"Iteration {i}")

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# As the number of iterations increases, the probability of success increases as well, but be careful not to overdo
# it or the results will start to get worse. This phenomenon is known as the "overcooking" of the state and is a
# consequence of rotating the state too much.
#
#
#
# Fixed-point Quantum Search
# --------------------------
# In the above example, we have a problem, we do not know the number of solutions that exist. Because of this we cannot
# calculate the number of iterations needed so it seems complicated to avoid overcooking. However, there is a variant
# of Amplitude Amplification that solve this issue, the Fixed-point quantum search variant [#fixedpoint]_.
#
# The idea behind this technique is to gradually reduce the intensity of the rotation we perform in the algorithm with
# the help of an auxiliary qubit.
# In this way, we will avoid rotating too much. The speed at which we decrease this intensity is carefully studied
# in the paper and has a very interesting interpretation related to the approximation of the
# sign function [#unification]_.
#
# To use this varaint we simply set ``fixed_point = True`` and indicate the auxiliary qubit.
# Let's see what happens with the same example as before:

@qml.qnode(dev)
def circuit(iters):

    U(wires=range(n))

    qml.AmplitudeAmplification(U = U(wires = range(n)),
                               O = oracle(range(n), range(n, n + 5)),
                               iters = iters,
                               fixed_point=True,
                               work_wire = n + 5)

    return qml.probs(wires = range(n))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i in range(1,9,2):
    output = circuit(iters=i)
    ax = axs[i // 4, i //2 % 2]
    ax.bar(range(len(output)), output)
    ax.set_ylim(0, 0.6)
    ax.set_title(f"Iteration {i}")

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# Unlike before, we can see that the probability of success does not decrease.
#
# Conclusion
# -----------
#
# In this demo we have shown the process of finding unknown states with Amplitude Amplification.
# We presented some of its limitations and learned how to overcome them with the Fixed-point version.
# The PennyLane template also helps you with other variants such as Oblivious Amplitude Amplification [#oblivious]_.
# We encourage you to explore these variants and see how they can help you in your quantum algorithms.
#
#
# References
# ----------
#
# .. [#ampamp]
#
#     Gilles Brassard, Peter Hoyer, Michele Mosca and Alain Tapp
#     "Quantum Amplitude Amplification and Estimation",
#     `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`__ (2000)
#
# .. [#fixedpoint]
#
#     Theodore J. Yoder, Guang Hao Low and Isaac L. Chuang
#     "Fixed-point quantum search with an optimal number of queries",
#     `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`__ (2014)
#
# .. [#unification]
#
#    John M. Martyn, Zane M. Rossi, Andrew K. Tan, Isaac L. Chuang.
#    “A Grand Unification of Quantum Algorithms”
#    `PRX Quantum 2,040203 <https://arxiv.org/abs/2105.02859>`__\ , 2021.
#
# .. [#oblivious]
#
#    Dominic W. Berry, et al.
#    "Simulating Hamiltonian dynamics with a truncated Taylor series",
#    `arXiv:1412.4687 <https://arxiv.org/pdf/1412.4687.pdf>`__, 2014
#
# About the author
# ----------------

