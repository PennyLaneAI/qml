r"""Intro to Amplitude Amplification
=======================================================================

`Grover's algorithm <https://pennylane.ai/qml/demos/tutorial_grovers_algorithm/>`_ is one of the most important algorithms developed in quantum computing. This technique belongs to a
much broader category of algorithms called Amplitude Amplification. In this demo, we will make an introduction to the
general problem by seeing how the idea proposed by Grover can be generalized and we will solve some of its limitations
with fixed-point quantum search.

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/OGthumbnail_large_AmplitudeAmplification_2024-04-29.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Amplitude Amplification
-------------------------

Our goal is to prepare an unknown state :math:`|\phi\rangle` using some known property of that state.
A good first approach is to use a unitary :math:`U` to generate an initial state :math:`U|0\rangle := |\Psi\rangle`
that "contains" some amount of the target state :math:`|\phi\rangle`.
We could then, try to manipulate this initial state :math:`|\Psi\rangle` so that it approaches :math:`|\phi\rangle`.
Generically we can represent any state, and in particular :math:`|\Psi\rangle`, in the computational basis as:

.. math::
    |\Psi\rangle = \sum_i c_i |i\rangle \quad \text{where} \quad c_i \in \mathbb{C},

but we can find better representations. One choice is to make :math:`|\phi\rangle` an element of the basis. We can then write

.. math::
    |\Psi\rangle = \alpha |\phi\rangle + \beta |\phi^{\perp}\rangle,

where  :math:`|\phi^{\perp}\rangle` is some state orthogonal
to :math:`|\phi\rangle` and :math:`\alpha, \beta \in \mathbb{R}`. This allows us to represent 
:math:`|\Psi\rangle` in a two-dimensional space making it easier to manipulate the state. This representation is even simpler than a Bloch
sphere since the amplitudes are real numbers --- we can visualize all operations inside a circle:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp1.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)


The two axes correspond to the basis elements. We also represent the initial state :math:`|\Psi\rangle` which
forms an angle of :math:`\theta` degrees with the x-axis: angle that can be calculated as :math:`\arcsin(\alpha)`.

Our aim is to *amplify* the amplitude :math:`\alpha` to get closer
to :math:`|\phi\rangle`, hence the name Amplitude Amplification [#ampamp]_. We will try to find an operator that moves
the initial vector :math:`|\Psi\rangle` as close to :math:`|\phi\rangle` as possible.

Finding the operators
~~~~~~~~~~~~~~~~~~~~~

To obtain the state :math:`|\phi\rangle`, we could create a rotation operator in this subspace and then rotate the initial state counterclockwise
by :math:`\pi/2 -\theta`, where :math:`\theta` is the angle shown above.  However, we don't explicitly know :math:`|\phi\rangle`,
so it's unclear how this could be done.  This is where a great idea is born: what if instead of rotations we think of
reflections?

The main insight of the Amplitude Amplification algorithm is that there is a sequence of **two reflections** that helps
us in this task. The first one is the reflection with respect to :math:`|\phi^{\perp}\rangle` and the second one is the reflection with respect
to :math:`|\Psi\rangle`.

Let's go step by step. First we apply the reflection around :math:`|\phi^{\perp}\rangle`:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp2.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Reflections with respect to :math:`|\phi^{\perp}\rangle`.

After applying the first reflection we are moving away from :math:`|\phi\rangle`, why do that?
Sometimes it is necessary to take a step backward in order to take two steps
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
we know the operator :math:`U` that generates :math:`|\Psi\rangle`.

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp3.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Reflections with respect to :math:`|\Psi\rangle`.


These two reflections are equivalent to rotating the state by :math:`2\theta` degrees from its original position,
where :math:`\theta` is the angle that defines the initial state. To amplify the amplitude and
approach the target state, we perform this sequence of rotations multiple times. More precisely, we repeat them :math:`k` times:

.. math::
    k = \frac{\pi}{4 \theta}-\frac{1}{2}.

Amplitude Amplification in PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After looking at the theory, let's take a look at a practical example using PennyLane: solving the `zero-sum problem <https://en.wikipedia.org/wiki/Zero-sum_problem>``.
In this problem we are given a list of :math:`n` integers and our goal is to find the subsets of numbers
whose sum is :math:`0`. Let's define our list of integers:
"""


values = [1, -2, 3, 4, 5, -6]
n = len(values)

##############################################################################
# The subset :math:`[1,5,-6]` is a solution to our problem but finding all of the solutions is a surprisingly expensive task.
# We will use Amplitude Amplification to solve the problem.
# First we define a binary variable :math:`x_i` that takes the value :math:`1` if we include the :math:`i`-th element in the
# subset and :math:`0` otherwise.
# We encode the :math:`i`-th variable in the :math:`i`-th qubit of a quantum state, so for instance, :math:`|100011\rangle`
# represents the subset above. We can now define the state:
#
# .. math::
#   |\Psi\rangle = \frac{1}{\sqrt{2^n}}\sum_{i=0}^{2^n-1}|i\rangle.
#
# This is a uniform superposition of all possible subsets so the solution is also "contained"
# in :math:`|\Psi\rangle`. We can generate the state :math:`|\Psi\rangle` by applying Hadamard gates. Let's generate the state and visualize it.

import pennylane as qml
import matplotlib.pyplot as plt
plt.style.use('pennylane.drawer.plot')

@qml.prod
def U(wires):
    # create the generator U, such that U|0⟩ = |Ψ⟩
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
plt.xlabel("|i⟩")
plt.axhline(0, color='black', linewidth=1)
plt.show()



##############################################################################
# Initially, the probability of getting any basis state is the same.
# The next step is to mark those elements that satisfy our property --- that the sum of the subset is :math:`0`.
# To do this we create an operator that stores the sum of the selected subset in some auxiliary qubits.
# This will allow us to "mark" our searched states without even knowing what they are.
# The operator is defined as:
#
# .. math::
#     \text{Sum}|x\rangle|0\rangle = |x\rangle|\sum v_ix_i\rangle,
#
# where :math:`v_i` is the :math:`i`-th integer. For more details of how we build this operation take a
# look at `Basic arithmetic with the QFT <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics/>`_.
#

import numpy as np

def Sum(wires_subset, wires_sum):
    qml.QFT(wires = wires_sum)
    for i, value in enumerate(values):
        for j in range(len(wires_sum)):
            qml.CRZ(value * np.pi / (2 ** j), wires=[wires_subset[i], wires_sum[j]])
    qml.adjoint(qml.QFT)(wires=wires_sum)

##############################################################################
# Therefore, in order to create the reflection around :math:`|\phi^{\perp}\rangle`, which we call it the oracle, we apply the :math:`\text{Sum}` operator to the
# state and then flip the sign of those states whose sum is :math:`0`.
# This allows us to mark the searched elements. Then we apply the inverse of the sum to clean the auxiliary qubits
# and eliminate its effect on the state.
#

@qml.prod
def oracle(wires_subset, wires_sum):
    # Reflection on |ϕ⟂⟩
    Sum(wires_subset, wires_sum)
    qml.FlipSign(0, wires=wires_sum)
    qml.adjoint(Sum)(wires_subset, wires_sum)


@qml.qnode(dev)
def circuit():
    U(wires=range(n))                 # Generate initial state
    oracle(range(n), range(n, n+5))   # Apply the reflection on |ϕ⟂⟩
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
# property. The next step is to reflect on the :math:`|\Psi\rangle` state.
# This reflection operator can be built directly in PennyLane with :class:`~.pennylane.Reflection`.
# The final circuit is then:
#
# .. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/sum_zero.jpeg
#    :align: center
#    :width: 60%
#    :target: javascript:void(0)
#


@qml.qnode(dev)
def circuit():
    U(wires=range(n))                  # Generate initial state
    oracle(range(n), range(n, n + 5))  # Apply the reflection on |ϕ⟂⟩
    qml.Reflection(U(wires=range(n)))  # Reflect on |Ψ⟩
    return qml.state()

##############################################################################
# Let's now look at the state :math:`|\Psi\rangle` and see how it is changed.

output = circuit()[0::2 ** 5].real
plt.bar(range(len(output)), output)
plt.ylim(-0.4, 0.6)
plt.ylabel("Amplitude")
plt.xlabel("|x⟩")
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# We have now amplified the amplitude of all the states that represent a solution to our problem.
# The four peaks are obtained in :math:`0`, :math:`27`, :math:`35` and :math:`61`, whose binary
# representation corresponds with :math:`|000000\rangle`, :math:`|011011\rangle`, :math:`|100011\rangle` and :math:`|111101\rangle` respectively.
# These states satisfy the property that the sum of the subset is :math:`0`.
#
# The Amplitude Amplification algorithm is implemented in PennyLane as :class:`~.AmplitudeAmplification`.
# Let's use this template to see the evolution
# of the state after multiple applications of the reflection operators.

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
# We can see that as the number of iterations increases, the probability of success increases as well. But we should be careful to not overdo
# the iterations or the results will start to get worse, as you can see for 5 iterations in our example. This phenomenon is known as "overcooking" the state and is a
# consequence of rotating the state too much.
#
# .. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/overcook.gif
#    :align: center
#    :width: 50%
#
#
#
# Fixed-point Quantum Search
# --------------------------
# In the above example, we have a problem: we do not know the number of solutions that exist. Because of this we cannot
# calculate the exact number of iterations that we need, so we do not know when to stop and might overcook the state. However, there is a variant
# of Amplitude Amplification that solves this issue: the fixed-point quantum search variant [#fixedpoint]_.
#
# The idea behind this technique is to gradually reduce the intensity of the rotation we perform in the algorithm with
# the help of an auxiliary qubit.
# In this way, we will avoid rotating too much. The speed at which we decrease this intensity is carefully studied
# in reference [#fixedpoint]_ and has a very interesting interpretation related to the approximation of the
# sign function [#unification]_.
#
# To use this variant we simply set ``fixed_point = True`` and indicate the auxiliary qubit.
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
# Unlike before, we can see that the probability of success does not decrease for a large number of iterations.
#
# Conclusion
# -----------
#
# In this demo we have shown the process of finding unknown states with Amplitude Amplification.
# We discussed some of its limitations and learned how to overcome them with the fixed-point version.
# This algorithm is important in different workflows such as molecular simulation with QPE. Moreover, the idea of using
# the reflections can be extrapolated to algorithms such as Qubitization or QSVT.
# PennyLane supports the Amplitude Amplification algorithm and its variants such as fixed-point and Oblivious Amplitude Amplification [#oblivious]_.
# We encourage you to explore them and see how they can help you in your quantum algorithms.
#
#
# References
# ----------
#
# .. [#ampamp]
#
#     Gilles Brassard, Peter Hoyer, Michele Mosca and Alain Tapp.
#     "Quantum Amplitude Amplification and Estimation",
#     `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`__, 2000.
#
# .. [#fixedpoint]
#
#     Theodore J. Yoder, Guang Hao Low and Isaac L. Chuang.
#     "Fixed-point quantum search with an optimal number of queries",
#     `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`__, 2014.
#
# .. [#unification]
#
#    John M. Martyn, Zane M. Rossi, Andrew K. Tan, Isaac L. Chuang.
#    “A Grand Unification of Quantum Algorithms”,
#    `PRX Quantum 2,040203 <https://arxiv.org/abs/2105.02859>`__, 2021.
#
# .. [#oblivious]
#
#    Dominic W. Berry, et al.
#    "Simulating Hamiltonian dynamics with a truncated Taylor series",
#    `arXiv:1412.4687 <https://arxiv.org/abs/1412.4687>`__, 2014
#
# About the author
# ----------------

