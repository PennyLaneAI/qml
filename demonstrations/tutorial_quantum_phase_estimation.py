r"""Tour of Quantum Phase Estimation
=============================================================

The Quantum Phase Estimation (QPE) algorithm is one of the most fundamental algorithms in quantum
computing. It is also one of the first algorithms that we expect to be practical in the Intermediate
Scale Quantum (ISQ) era. The algorithm solves a relatively simple task: finding the eigenvalue of a
unitary operator. Solving this problem efficiently has pivotal applications in many areas of science
such as calculating molecular energies in chemistry, solving linear system of equations, and quantum
counting. This demo explains the QPE algorithm and gives an intuition to help us exploit its full potential.

.. figure:: ../_static/demonstration_assets/quantum_phase_estimation/socialthumbnail_large_Quantum_Phase_Estimation_2023-11-27.png
    :align: center
    :width: 60%
    :target: javascript:void(0)


The problem
-----------

Let's first get a better understanding of the problem we are trying to solve. We are given a unitary
:math:`U` operator and one of its eigenvectors :math:`|\psi \rangle`. For a unitary operator, we
know that:

.. math::
    U |\psi \rangle = e^{2 \pi i \theta} |\psi \rangle,

where :math:`\theta` is called the *phase* of the eigenvalue. Our task is to find the *phase* and
the QPE algorithm helps us to approximate the value of the phase on a quantum computer. That is why
the algorithm is called quantum phase estimation! But how a quantum computer can solve this problem
better than classical computers?

There are several cases where a quantum computer outperforms the best known classical algorithm for
solving a problem. The most famous example is the Shor's algorithm for factorizing a prime number.
What Peter Shor did was to transform this problem into a problem that we know how to solve more
efficiently on a quantum computer: calculating the period of a function.

The QPE algorithm can be understood based on the same idea: it translates the phase search problem
into the calculation of the period of a function. To understand how, let's first see how we can
calculate the period of a function classically.

Calculating the period
----------------------

An elegant way to compute the period of a function is to use the famous method of
`Fourier Transform <https://en.wikipedia.org/wiki/Fourier_transform>`_. To do this for a function
:math:`g(x)`, we evaluate the function for :math:`N` different values of :math:`x` and generate the
vector:

.. math::
    \vec{v} = (g(x_0), g(x_1), \dots, g(x_{N-1})).

Applying the Fourier transform to this vector gives us a new vector that contains information about
the frequency, and hence the period, of the original function. This is a technique widely used in
signal processing.

Let's see an example. We chose the periodic function :math:`g(x) = e^{\frac{\pi i x}{5}}` with the
period :math:`T = 10`. The :math:`x_i` can be simply chosen as integers from :math:`0` to
:math:`31`. This is how the function and its fourier transform look like:
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

xs = np.arange(0, 32, 1)
f_xs = np.exp(np.pi * 1j * xs / 5)

axs[0].plot(xs, f_xs.real, label = "real part")
axs[0].plot(xs, f_xs.imag, label = "imaginary part")
axs[0].set_title('Exp function')
axs[0].set_xlabel('x')
axs[0].set_ylabel('g(x)')
axs[0].legend()

# We use the numpy implementation of the Fourier transform
ft_result = np.abs(np.fft.fft(f_xs))

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
# can now be computed as:
#
# .. math::
#     T = \frac{N}{f_0}.
#
# In our particular example with :math:`N = 32` and :math:`f_0 = 3`, the period is
# :math:`T \approx 10.67` which is very close to the exact value :math:`10` 🎉.
#
# The QPE algorithm
# -----------------
# The QPE algorithm does something similar to what we saw above: it helps us to find the period
# :math:`T = \frac{1}{\theta}` of the function :math:`g(x) = e^{2 \pi i \theta x}` where
# :math:`e^{2 \pi i \theta}` is the eigenvalue of our desired unitary operator. To implement the
# algorithms, we first need the vector:
#
# .. math::
#    \vec{v} = (g(x_0), g(x_1), \dots, g(x_{N-1})) =
#    (e^{2 \pi i \theta (0)}, e^{2 \pi i \theta (1)}, \dots, e^{2 \pi i \theta (N-1)}).
#
# We can represent this vector by a state vector on a quantum computer. The desired state vector
# can be constructed by applying a sequence of controlled unitary gates raised to increasing powers
# of math:`2`. Let's look an example for :math:`N = 8`. For simplicity, we first construct the
# vector :math:`(0, 0, 0, 0, 0, 0, e^{2\pi i \theta (6)}, 0, 0)`. The following image illustrates
# the circuit that creates this state.
#
# .. figure::
#   ../_static/demonstration_assets/quantum_phase_estimation/controlled_sequence2.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# This approach can be easily generalized to create our desired vector
# math:`(e^{2 \pi i \theta (0)}, e^{2 \pi i \theta (1)}, \dots, e^{2 \pi i \theta (N-1)})`. We just
# need to start from :math:`N^{(-1/2)} (1, 1, 1, 1, 1, 1, 1, 1, 1)` instead of :math:`|6 \rangle` in
# our example. This can be efficiently constructed by applying Hadamard gates to our qubits
# initialized at a :math:`|0 \rangle` state. After having the desired state, all we need to do is to
# apply the adjoint of the quantum Fourier transform which encodes the phase into the state of our qubits.
# But how a number such as math:`\theta = 0.532` can be encoded into the states? This can be done by
# using a nice trick: representing math:`\theta` in its binary format. For example, in the case of
# our math:`3` qubit circuit, a measured state of
# math:`|0 \rangle \otimes |1 \rangle \otimes |1 \rangle` corresponds to math:`\overline{0.001}`
# where the bar denotes binary representation. Let's convert this binary number to decimal.
#
# In the same way that a decimal number such as math:`0.125` can be written as
#
# .. math::
#    0.125 = \frac{1}{10^1} + \frac{2}{10^2} + \frac{5}{10^3},
#
# we can write the binary number math:`\overline{0.001}` as
#
# .. math::
#    \overline{0.001} = \frac{0}{2^1} + \frac{0}{2^2} + \frac{1}{2^3} = 0 + 0 + \frac{1}{8} = 0.125
#
# Here you go 🧠!
#
# A very interesting fact about binary numbers is that, again, in the same way that multiplying
# math:`0.125` by math:`10` shifts the decimal point by one difit to the right, multiplying
# math:`\overline{0.001}` by math:`2` gives math:`\overline{0.01}`. Try to verify it. This is very
# important for QPE because by applying the powers of the unitary operator we practically shift the
# decimal point in our phase to the left:
#
# .. math::
#    U^2 |\psi \rangle = e^{ (2 \pi i) \overline{\theta_1 . \theta_2 \theta_3}} |\psi \rangle = e^{\overline{0. \theta_2 \theta_3}} |\psi \rangle.
#
# Note that math:`\overline{\theta_1}` is either math:`0` or math:`1` which gives
# math:`e^{(2 \pi i) \overline{\theta_1}} = 1`. By applying math:`U` with increasing powers of
# math:`2` we practically encode different number of digits of our phase in the quantum state of
# each qubit. Then the adjoint of the quantum Fourier transform transforms the quantum state to
# math:`|\theta_1 \rangle \otimes |\theta_2 \rangle \otimes |\theta_3 \rangle`. The overall circuit
# for applying QPE looks like:
#
# .. figure::
#   ../_static/demonstration_assets/quantum_phase_estimation/phase_estimation.png
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# .. note::
#     By definition the classical Fourier transform coincides with the inverse of the quantum
#     Fourier transform.
#
# Time to code!
# -------------
# Great, we already know the three building blocks of QPE and it is time to put them in practice.
# We use the operator :math:`U = R_{\phi}(2 \pi / 5)` and its eigenvector :math:`|1\rangle``.
#

import pennylane as qml

def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires = wires)

estimation_wires = [1, 2, 3]

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |1>
    qml.PauliX(wires=0)

    # We create the normalized vector (1, 1, ..., 1, 1)
    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    # We apply the function g to all values
    qml.ControlledSequence(U(wires = 0), control=estimation_wires)

    # We apply the inverse QFT to obtain the frequency
    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)


results = circuit_qpe()
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.show()

##############################################################################
# Similar to the classical case, we have the frequencies and their magnitude. The peak of the
# frequency is at the value :math:`2` and knowing that :math:`N = 8` we have :math:`T = 4`.
# Our approximation of :math:`\theta` is :math:`1/4 = 0.25`, close to the exact value of
# :math:`0.2`. You can increase the number of estimation qubits to see how the approximation
# improves.
#
# Cleaning the signal
# -------------------------
# One of the advantages of the relationship between classical signal processing and QPE is that we can reuse knowledge
# and that is something we are going to do to improve our output. As we can see, the plot above shows with a small probability
# some noisy frequencies.
# These tails are called *leaks* and cause us to obtain unwanted values.
# One of the most commonly used techniques in signal processing is the use of windows, which are a type of function
# that is applied to the initial vector before apply the Fourier Transform. An example is to use the cosine
# window  [#Gumaro]_. In QPE the window refers to the initial block, so we simply replace the Hadamard with the
# :class:`~.CosineWindow` operator.
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
# Goodbye *leaks*! As you can see a small modification in the algorithm has *filtered* the noisy that were generated.
# Furthermore, such an operator can be efficiently constructed on a quantum computer!
#
# Conclusion
# ----------
# In this demo we have seen what is Quantum Phase Estimation and how it relates with signal processing.
# The great advantage of this approach we have shown is that it creates a perfect bridge between the classical
# the quantum techniques. Therefore, future very important lines of research can be oriented to translate all the
# discoveries already made in classical into the quantum field.
# I invite you to experiment with other examples and demonstrate what you have learned!
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
