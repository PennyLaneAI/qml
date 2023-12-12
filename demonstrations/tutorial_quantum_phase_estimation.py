r"""Intro to Quantum Phase Estimation
=============================================================

The Quantum Phase Estimation (QPE) algorithm is one of the most fundamental tools in quantum
computing. The algorithm solves a relatively simple task: finding eigenvalues of a
unitary operator. Solving this problem efficiently has important applications in many areas of science
such as calculating molecular energies in chemistry, solving linear system of equations, and quantum
counting. This demo explains the QPE algorithm and gives an intuition to help us exploit its full potential.

.. figure:: ../_static/demonstration_assets/quantum_phase_estimation/socialthumbnail_large_Quantum_Phase_Estimation_2023-11-27.png
    :align: center
    :width: 30%
    :target: javascript:void(0)


The problem
-----------

Let's first get a better understanding of the problem we are trying to solve. We are given a unitary
operator :math:`U`  and one of its eigenvectors :math:`|\psi \rangle`. For a unitary operator, we
know that:

.. math::
    U |\psi \rangle = e^{i \theta} |\psi \rangle,

where :math:`\theta` is the *phase* of the eigenvalue and has a value between :math:`0` and
:math:`2 \pi`. The QPE algorithm helps us to estimate the value of the phase on a quantum
computer --- hence its name. But how does it work?

There are many cases where a quantum computer outperforms the best known classical algorithm for
solving a problem. Arguably the most famous example is Shor's factoring algorithm, which works by
transforming the factoring problem into a task that we know how to solve more efficiently on a
quantum computer: calculating the period of a function. The QPE algorithm can be understood based
on the same idea: it translates the phase search problem into the calculation of the period of a
function. To understand how, let's first see how we can do this classically.

Calculating the period
----------------------

An elegant way to compute the period of a function is to use a
`Fourier transform <https://en.wikipedia.org/wiki/Fourier_transform>`_. To do this for a function
:math:`g(x)`, we evaluate the function for :math:`N` different values of :math:`x` and generate the
vector

.. math::
    \vec{v} = [g(x_0), g(x_1), \dots, g(x_{N-1})].

Applying the Fourier transform gives us a new vector that contains information about the frequency,
and hence the period, of the original function. This is a technique widely used in signal
processing. Recall that the period of a function, :math:`T`, denotes the distance between the
repetitions of a periodic function. The period is inversely related to the frequency, :math:`f`,
which is number of repetitions in a defined range :math:`N` as

.. math::
    T = \frac{N}{f}.

Let's see an example. We chose the periodic function :math:`g(x) = e^{\frac{\pi i x}{5}}`, which has
a period :math:`T = 10`. (Can you see why the period is 10?) The :math:`x_i` can be simply chosen as
integers from :math:`0` to :math:`31`.
"""

import numpy as np
import matplotlib.pyplot as plt

xs = np.arange(0, 32, 1)
f_xs = np.exp(np.pi * 1j * xs / 5)

# We use the numpy implementation of the Fourier transform
ft_result = np.abs(np.fft.fft(f_xs))

##############################################################################
# This is how the function and its fourier transform look like:

plt.style.use('pennylane.drawer.plot')
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(xs, f_xs.real, label = "real part")
axs[0].plot(xs, f_xs.imag, label = "imaginary part")
axs[0].set_title('Exp function')
axs[0].set_xlabel('x')
axs[0].set_ylabel('g(x)')
axs[0].legend()

axs[1].bar(xs, ft_result)
axs[1].set_title('Fourier Tranform')
axs[1].set_xlabel('Frecuency')
axs[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

##############################################################################
# Note that we have separated the real and imaginary parts to better visualize the function. The
# right graph shows the possible frequencies and their magnitude. We use the value with the largest
# magnitude to approximate the fundamental frequency, i.e., :math:`f_0 = 3`. The period :math:`T`
# can now be computed as :math:`T = \frac{N}{f_0}`. In our particular example with :math:`N = 32`
# and :math:`f_0 = 3`, the period is :math:`T \approx 10.67` which is very close to the exact value
# :math:`10` 🎉.
#
# The QPE algorithm
# -----------------
# The quantum phase estimation algorithm finds the eigenvalue of a unitary operator by estimating
# the period of the function :math:`g(x) = e^{2 \pi i \theta x}`. The algorithm has the following
# main steps.
#
# 1. Create a uniform superposition by applying Hadamard gates to our qubits initialized at a
# :math:`|0 \rangle` state.
#
# 2. Apply a sequence of controlled unitary gates raised to increasing powers of :math:`2`. This
# allows us to encode :math:`g(x)` as amplitudes of the quantum state.
#
# 3. Apply the adjoint of the quantum Fourier transform. This encodes the binary representation of
# the phase into the state of our qubits.
#
# .. note::
#     By definition the classical Fourier Transform coincides with the inverse of the Quantum Fourier
#     Transform, that is why the adjoint is put.
#
# Then the overall circuit for applying QPE with these three main blocks is shown below.
#
# .. figure::
#   ../_static/demonstration_assets/quantum_phase_estimation/phase_estimation.png
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# .. note::
#     We can encode a number into a quantum state by representing it in binary. In the same way
#     that a decimal number such as :math:`0.125` can be written as
#
#     .. math::
#        0.125 = \frac{1}{10^1} + \frac{2}{10^2} + \frac{5}{10^3},
#
#     we can write the binary number :math:`\overline{0.001}` as
#
#     .. math::
#        \overline{0.001} = \frac{0}{2^1} + \frac{0}{2^2} + \frac{1}{2^3} = 0 + 0 + \frac{1}{8} = 0.125
#
#     The same way that multiplying :math:`0.125` by :math:`10` shifts the decimal point by one
#     digit to the right, multiplying :math:`\overline{0.001}` by :math:`2` gives
#     :math:`\overline{0.01}`. Try to verify it.
#
# Time to code!
# -------------
# Great, we already know the three building blocks of QPE and it is time to put them in practice.
# We use the operator :math:`U = R_{\phi}(2 \pi / 5)` and its eigenvector :math:`|1\rangle`` to
# estimate the eigenvalue of :math:`U`.
#

import pennylane as qml

def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires = wires)

##############################################################################
# We then construct a uniform superposition by applying Hadamard gates and apply controlled unitary
# gates with :class:`~.ControlledSequence`. Finally, we perform the adoint of :class:`~.QFT` and
# return the probability of each computational basis state. We use five ancilla qubits to get a
# binary representation of the phase with five digits.


estimation_wires = range(1, 9)

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |1>
    qml.PauliX(wires=0)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires = 0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

##############################################################################
# Let's run the circuit and plot the results.

results = circuit_qpe()
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.show()

##############################################################################
# Similar to the classical case, we have the frequencies and their magnitude. The peak of the
# frequency is at the value :math:`51` and knowing that :math:`N = 256`, our approximation of
# :math:`\theta` is :math:`0.19921875`, close to the exact value of :math:`0.2`.
#
# Increasing the number of estimation qubits improves the approximation of the phase. The reason is
# very simple: if we have :math:`3` estimation qubits, the best number we get is a :math:`3`-digit
# number among the possible combinations :math:`[000, 100, 010, 001, 110, 101, 011, 111]`. However,
# if we have more and more qubits, our space of choice becomes larger and the number of digits
# increases. This makes the estimated number closer and closer to the exact value.
#
# Cleaning the signal
# -------------------
# The plot we obtained above is noisy and has several unwanted frequencies called *leaks*. Here we
# borrow a technique from classical signal processing to improve our output. One of the most
# commonly used techniques in signal processing is the use of windows. These are functions that are
# applied to the initial vector before applying the Fourier transform. We use a similar method here
# and apply a cosine window [#Gumaro]_ to the initial block, so we simply replace the Hadamard gates
# with the :class:`~.CosineWindow` operator. This operator can be efficiently constructed on a
# quantum computer!
#

@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |1>
    qml.PauliX(wires=0)

    # We apply the window
    qml.CosineWindow(wires = estimation_wires)

    # We apply the function f to all values
    qml.ControlledSequence(U(wires = 0), control=estimation_wires)

    # We apply the inverse QFT to obtain the frequency
    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)


results = circuit_qpe()
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.show()

##############################################################################
# Goodbye *leaks*! As you can see, a small modification in the algorithm has *filtered* the noise.
#
# Conclusion
# ----------
# This demo provides a brief introduction to quantum phase estimation. We learned that the QPE
# algorithm works by encoding the eigenvalue of a unitary operator into the quantum state of a set
# of ancilla qubits. Measurements in the computational bases, will then give us a binary
# representation of the eigenvalues. The number of digits in this binary representation determines
# the accuracy of the final results such that having more ancilla qubits gives us a more accurate
# estimation. The demo exploits the relation between quantum phase estimation and signal processing
# to build a bridge between the classical the quantum techniques. This might help to translate
# classical discoveries into the quantum field. You can experiment with other examples to
# demonstrate what you have learned!
#
# References
# ---------------
#
# .. [#Gumaro]
#
#     Gumaro Rendon, Taku Izubuchi, Yuta Kikuchi,  `"Effects of Cosine Tapering Window on Quantum Phase Estimation" <https://arxiv.org/abs/2110.09590>`__.
#
#
# About the author
# ----------------
#
