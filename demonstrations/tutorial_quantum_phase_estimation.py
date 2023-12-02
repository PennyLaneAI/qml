r"""Tour of Quantum Phase Estimation
=============================================================

One of the first algorithms we expect to be able to run as we move into the ISQ era is Quantum Phase Estimation (QPE).
The aim of this demo will be to understand this algorithm and give an intuition that will help us to exploit its
full potential.

.. figure:: ../demonstrations/quantum_phase_estimation/socialthumbnail_large_Quantum_Phase_Estimation_2023-11-21.png
    :align: center
    :width: 60%
    :target: javascript:void(0)


Presentation and motivation of the problem
-----------------------------------------

The first thing is to understand a little better the problem we are trying to solve. Given a unitary :math:`U`,
and one of its eigenvectors :math:`|\psi \rangle`, we know there is a :math:`\theta` such as:

.. math::
    U |\psi \rangle = e^{2 \pi i \theta} |\psi \rangle.

Quantum Phase Estimation is an algorithm that allow us to approximate that :math:`\theta` phase.

So, the goal is clear but, why is a quantum computer supposed to solve this task better?
There are really few applications in which it has been demonstrated that a quantum computer actually outperforms
the best classical algorithm. The most famous example of this is Shor's algorithm. What Peter Shor did was to
transform a problem of interest - the factorization of prime numbers - into a problem that we know that a quantum
computer is more efficient: the calculation of the period of functions.

As it turns out, that strategy makes a lot of sense so we are going to imitate it! We will translate the problem of
finding this phase into the problem of finding the period of a function.

Calculation of the period classically
---------------------------------------

Calculating the period of a function :math:`f` is something that can be done classically with the help of the Fourier
Transform. To do this, we evaluate the function consecutively :math:`N` times and generate the vector:

.. math::
    \vec{v} = (f(x_0), f(x_1), \dots, f(x_{N-1})).

After applying the Fourier transform to this vector we will obtain a new vector that will give us information of the
frequency of the original function. Let's see an example for the function :math:`f(x) = cos(\frac{\pi x}{5})` whose
period is :math:`T \approx 10.67`. We will choose the :math:`x_i` as integer values from :math:`0` to :math:`31`.
The function would look like this:
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

xs = np.arange(0, 32, 1)
f_xs = np.cos(np.pi * xs / 5)

plt.plot(xs, f_xs)

plt.show()

##############################################################################
#
# Let's see now what happens when we apply the Fourier Transform to it.
#

# We apply the fourier transform provided by numpy
ft_result = np.abs(np.fft.fft(f_xs))

# Let's plot the first 30 elements of the result
plt.bar(xs, ft_result)
plt.xlabel("frequency")
plt.show()


##############################################################################
# This graph shows on the x-axis the value of the possible frequencies and on the y-axis the magnitude of their
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
# Quantum Phase Estimation, will make use of this with the above reasoning to be able to find the :math:`\theta` phase
# we were looking for. Suppose we had the function :math:`f(x) = e^{2 \pi i \theta x}`. This is also a periodic function
# with :math:T` = \frac{1}{\theta}`. Therefore, if we were able to obtain the vector:
#
# .. math::
#    \vec{v} = (f(x_0), f(x_1), \dots, f(x_{N-1})) = (e^{2 \pi i \theta 0}, e^{2 \pi i \theta 1}, \cdot, e^{2 \pi i \theta (N-1)}),
#
# following the previous idea, we could obtain the period and with it :math:`\theta`.
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
# The construction of the vector is done in two stages. The central block, which we will call ControlSequence, is in
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
# Seen in this way, if we want to construct :math:`v`, we simply need to send to SequenceControl the vector :math:`(1,1,\dots, 1)`
# to store the value of the function in all points. Being working in a quantum computer the vector will be normalized
# by a factor of :math:`\frac{1}{\sqrt{N}}` and this can be efficiently constructed by simply applying Hadamard gates.
# Hence the reason for the initial block!
#
# Time to code!
# -----------------
#
# Great, we already know the meaning of each QPE block so it's time to put it into practice.
# For this purpose, we can take as operator :math:`U = R_{phi}(2 * pi / 5)` and as eigenvector :math:`|1\rangle``
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
# The measurement problem
# -------------------------
#
# In the previous explanation, we have been obtaining the fundamental frequency visually by observing where the peak was.
# However, when using quantum computers we will not have access to that exact distribution, but we will try to
# approximate it from a series of shots. We will take that our fundamental frequency is the average of the frequencies
# obtained.

np.random.seed(42)
dev2 = qml.device("default.qubit", shots = 20)

@qml.qnode(dev2)
def circuit_qpe():

    qml.PauliX(wires=0)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires = 0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.sample(wires=estimation_wires)


samples = circuit_qpe()
estimated_f0 = np.mean([int(''.join(str(bit) for bit in binary_list), 2) for binary_list in samples])
print("θ approximation:", 1/(8 / estimated_f0))

##############################################################################
# As we can see, by limiting ourselves to :math:`20` shots, the approximation to :math:`\theta` is further away from
# the value :math:`\theta = 0.2`. One of the reasons for this is the concept of leakage.
# Our frequency spectrum shown above had certain values far from the correct one with a certain probability.
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/leakage.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# These tails are called leaks and cause us to obtain unwanted values.
# Thanks to the interpretation we have given of the algorithm, relating it to classical signal processing,
# we can use existing techniques in this field to solve this problem.
# One of the most commonly used techniques is the use of windows, which are a type of function that is applied to the
# initial state. An example is to use the cosine window  [#Gumaro]_, instead of the use of Hadamard gates.
# Let's see what the new generated distribution would look like:
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
# Goodbye noise! Let's see how close we can get now using 20 shots.

@qml.qnode(dev2)
def circuit_qpe():

    qml.PauliX(wires=0)

    qml.CosineWindow(wires = estimation_wires)

    qml.ControlledSequence(U(wires = 0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.sample(wires=estimation_wires)


samples = circuit_qpe()
estimated_f0 = np.mean([int(''.join(str(bit) for bit in binary_list), 2) for binary_list in samples])
print("θ approximation:", 1/(8 / estimated_f0))

##############################################################################
# Great! So a small modification in the algorithm has mitigated the errors that were generated.
#
# Conclusion
# ----------
# In this demo we have seen how Quantum Phase Estimation works relating it to signal processing.
# The great advantage of this approach we have shown you is that it creates a perfect bridge between the classical
# the quantum techniques. Therefore, future very important lines of research can be oriented to translate all the
# discoveries already made in classical signal processing  into the quantum field.
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
