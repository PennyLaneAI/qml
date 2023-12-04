r"""Tour of Quantum Phase Estimation
=============================================================

One of the first algorithms we expect to be able to run as we move into the ISQ era is *Quantum Phase Estimation* (QPE).
This algorithm solves a simple task that has many applications such as calculating energies in chemistry, solving linear
system of equations or the quantum counting subroutine.

The aim of this demo will be to explain this algorithm and give an intuition that will help us to exploit its
full potential.

.. figure:: ../demonstrations/quantum_phase_estimation/socialthumbnail_large_Quantum_Phase_Estimation_2023-11-27.png
    :align: center
    :width: 60%
    :target: javascript:void(0)


Presentation and motivation of the problem
-----------------------------------------

The first thing is to understand a little better the problem we are trying to solve. We are given a unitary :math:`U`,
and one of its eigenvectors :math:`|\psi \rangle`. As a unitary operator we know that there is a :math:`\theta` such that:

.. math::
    U |\psi \rangle = e^{2 \pi i \theta} |\psi \rangle.

This :math:`\theta` value is called the *phase* of the eigenvalue and is the element we will try to calculate.
*Quantum Phase Estimation* is one of the most relevant techniques to approximate this value on a quantum computer.
But, why is a quantum computer supposed to solve this task better?

There are few applications in which it has been demonstrated that a quantum computer actually outperforms
the best classical algorithm. The most famous example that achieves exponential advantage is Shor's algorithm.
What Peter Shor did was to transform a problem of interest - the factorization of prime numbers - into a problem that
we know that a quantum computer is more efficient: the calculation of the period of functions.

QPE manages to exploit the same idea: it translates the phase search problem into the calculation of the
period of a given function. To understand how, we must begin by answering a question first.
How can we calculate the period of a function classically?

Calculation of the period classically
---------------------------------------

Calculating the period of a function :math:`f` is something that can be done with the help of the Fourier
Transform. To do this, we evaluate the function consecutively :math:`N` times and generate the vector:

.. math::
    \vec{v} = (f(x_0), f(x_1), \dots, f(x_{N-1})).

After applying the Fourier transform to this vector we will obtain a new vector that will give us information of the
frequency of the original function. This is a technique widely used in signal processing. Let's see an example for the
function :math:`f(x) = \cos(\frac{\pi x}{5})` whose period is :math:`T = 10`. We will choose the :math:`x_i` as integer
values from :math:`0` to :math:`31`. Let's see what the function and its fourier transform look like:
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

xs = np.arange(0, 32, 1)
f_xs = np.cos(np.pi * xs / 5)

axs[0].plot(xs, f_xs)
axs[0].set_title('Cosine function')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x)')

# We use the numpy implementation of the Fourier Transform
ft_result = np.abs(np.fft.fft(f_xs))

axs[1].bar(xs, ft_result)
axs[1].set_title('Fourier Tranform')
axs[1].set_xlabel('Frecuency')
axs[1].set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

##############################################################################
# The right graph shows on the x-axis the value of the possible frequencies and on the y-axis the magnitude of their
# relevance to the initial function. Focusing on the first half, we can see a peak at :math:`3`.
# A simple approximation is to take the value of said peak as the fundamental frequency.
# If we now want to get the period :math:`T`, we must apply:
#
# .. math::
#     T = \frac{N}{f_0},
#
# where :math:`f_0` represents the detected fundamental frequency. In our particular example for :math:`N = 32`
# and :math:`f_0 = 3`, we obtain that the period is :math:`T \approx 10.67` that is very close to the real
# value :math:`10`.
#
# Similarity to QPE
# -------------------
# The Fourier Transform is something we can also run on a quantum computer through the QFT operator.
# Quantum Phase Estimation will make use of this with the above reasoning to be able to find the :math:`\theta`
# we were looking for. Suppose we had the function :math:`f(x) = e^{2 \pi i \theta x}`. This is also a periodic function
# with :math:`T=\frac{1}{\theta}`. Therefore, if we were able to obtain the period, we will find the phase.
# To do that, the first step is to create the vector:
#
# .. math::
#    \vec{v} = (f(x_0), f(x_1), \dots, f(x_{N-1})) = (e^{2 \pi i \theta 0}, e^{2 \pi i \theta 1}, \dots, e^{2 \pi i \theta (N-1)}),
#
# In QPE we can find 3 fundamental blocks: an initial row of Hadamards, a sequence of control gates and the inverse of
# the QFT. The first two blocks will help us to construct the vector and finally we will apply the adjoint of the
# Fourier transform to recover :math:`\theta`.
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/phase_estimation.png
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# .. note::
#     By definition the classical Fourier Transform coincides with the inverse of the Quantum Fourier Transform,
#     that is why the adjoint is put.
#
# The construction of the vector is done in two stages. The central block, which we will call :class:`~.ControlledSequence`, is in
# charge of evaluating the function itself. It works as follows: if we send it the vector :math:`(0,0,\dots,1,\dots,0,0)`,
# with just one 1 in the j-th position, we will obtain the vector :math:`(0,0,\dots,e^{2\pi i \theta j},\dots,0,0)`.
# To understand this, let's take a look at the following image:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/controlled_sequence2.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# The input vector that has only one :math:`1` is easily encoded in binary. In the example we have taken :math:`j = 6`,
# which can be encoded as :math:`|110\rangle`. The powers of the :math:`U` operators use precisely the binary
# representation to encode the phase. As we can see, we are adding the corresponding value inside the vector.
#
# Seen in this way, if we want to construct :math:`v`, we simply need to send to the Controlled Sequence the vector :math:`(1,1,\dots, 1)`
# to store the value of the function in all points. Being working in a quantum computer the vector will be normalized
# by a factor of :math:`\frac{1}{\sqrt{N}}` and this can be efficiently constructed by simply applying Hadamard gates.
# Hence the reason for the initial block!
#
# Time to code!
# -----------------
#
# Great, we already know the meaning of each QPE block so it's time to put it into practice.
# For this purpose, we can take as operator :math:`U = R_{\phi}(2 \pi / 5)` and as eigenvector :math:`|1\rangle``
#

import pennylane as qml

def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires = wires)

estimation_wires = [2, 3, 4]

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |1>
    qml.PauliX(wires=0)

    # We create the vector (1,1,...,1,1) normalized
    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

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
#
# Similar to the classical reasoning, we are now looking at the values of the possible frequencies with their magnitude.
# The peak of the frequency is found at the value :math:`2` and knowing that :math:`N = 8` we would have
# that :math:`T = 4`. Therefore our approximation of :math:`\theta` will be :math:`1/4 = 0.25`, close to the real value
# of :math:`0.2`. I invite you to increase the number of estimation qubits to see how the approach improves.
#
# Cleaning the signal
# -------------------------
#
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
