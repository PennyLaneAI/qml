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
frequency of the original function. Let's see an example for the function :math:`f(x) = cos(\frac{\pi x}{5})' whose
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
# If we now want to get the period :math:`T`, we must apply:
#
# .. math::
#     T = \frac{N}{f_0},
#
# :math:`f_0` represents the detected fundamental frequency. In our particular example for :math:`N = 32`
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
#.. math::
#    \vec{v} = (f(x_0), f(x_1), \dots, f(x_{N-1})) = (e^{2 \pi i \theta 0}, e^{2 \pi i \theta 1}, \cdot, e^{2 \pi i \theta (N-1)}),
#
# following the previous idea, we could obtain the period and with it :math:`\theta`.
#
# In QPE we can find 3 fundamental blocks: an initial row of Hadamards, a sequence of control gates and the inverse of
# the QFT. The first two blocks will help us to construct the vector and finally we will apply the adjoint of the
# Fourier transform to recover :math:`\theta`.
#
# .. note::
#     By definition the classical Fourier Transform coincides with the inverse of the Quantum Fourier Transform,
#     that is why the adjoint is put.
#
# The construction of the vector is done in two stages. The central block, which we will call ControlSequence, is in
# charge of evaluating the function itself. It works as follows: if we send it the vector :math:`(0,0,\cdot,1,\cdot,0,0)`,
# with just one 1 in the j-th position, we will obtain the vector :math:`(0,0,\cdot,e^{2\pi i \theta j},\cdot,0,0)`.
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
# Seen in this way, if we want to construct :math:`v`, we simply need to send to SequenceControl the vector :math:`(1,1,\cdot, 1)`
# to store the value of the function in all points. Being working in a quantum computer the vector will be normalized
# by a factor of :math:`\frac{1}{\sqrt{N}}` and this can be efficiently constructed by simply applying Hadamard gates.
# Hence the reason for the initial block!

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
qml.draw_mpl(circuit_qpe)()
plt.show()

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
# Mitigating leakage
# -------------------
#
# Although we have so far presented the QPE algorithm in a more traditional way, there are small modifications that can help us solve certain problems. But what are the problems with the procedure presented above?
# In principle everything seems to work fine but we have been manually accessing the vector of our quantum state to obtain the frequency with higher probability. When it comes down to it, this is not something we will be able to do and we will have to obtain this information by sampling. In the previous example, we have seen some kind of tails in our state:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/leakage.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# These tails are called leaks and cause us to obtain unwanted values. In an ideal world in which we could work with all the points of our function these errors would not appear. However, having to work with a finite number of points we will have to deal with this situation. To give you an intuitive idea, although in all the previous section we have talked about calculating the period of the function :math:`f` in general, we are really limiting ourselves to an interval as seen in this image:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/leakage2.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# The function that we generate with the help of the quantum computer or in the classical method is truncated to an interval marked by :math:`h(x)`. Such a function will take the value :math:`1` if :math:`x` belongs to the interval we want to work on and :math:`0` if we are outside. In the quantum case, the size of our window will be :math:`[0,...,2^m-1]` where :math:`m` is the number of estimation wires. For this reason, increasing the number of qubits will improve the accuracy, because we are working on more points of the function.
#
# Now that we know that our input is really the function :math:`f(x) \cdot h(x)`, we can deduce where the error comes from when trying to calculate the frequency. It is that we are not applying the QFT to :math:`f` but to the product of those two functions. But here we can make use of a very interesting property of the Fourier transform and it is that:
#
# .. math::
#   \mathcal{F}(f \cdot h) = \mathcal{F}(f) \ast \mathcal{F}(h)
#
# where :math:`\ast` refers to convolution. This gives us a big clue to find suitable functions to work with. The best ones will be those whose Fourier transform approximates a delta function (since it is the identity for the convolution) and hence:
#
# .. math::
#   \mathcal{F}(f(x)\cdot h(x)) = \mathcal{F}(f(x)) \ast \mathcal{F}(h(x)) \approx  \mathcal{F}(f(x)).
#
# This is a field widely studied in signal theory from a classical point of view. Applying all Hadamard gates generates what is known as the `BoxCar window <https://en.wikipedia.org/wiki/Boxcar_function>`__. However now with Pennylane you have access to more advanced windows such as the :class:`~pennylane.CosineWindow`.
# Let's see an example to see if it works:

estimation_wires = [2, 3, 4, 5]

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |10>
    qml.PauliX(wires=0)

    # We apply the cosine window.
    qml.CosineWindow(wires=estimation_wires)

    # We apply the function f to all values

    qml.ControlledSequence(qml.TrotterProduct(A, time=1), control=estimation_wires)

    # We apply the inverse QFT to obtain the frequency

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)


circuit_qpe()
results = circuit_qpe()
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.ylabel("prob")
plt.show()

##############################################################################
# Great! A small modification in the previous code has mitigated the errors that were generated. Also the Cosine Window can be implemented efficiently on a quantum computer!
#
# Conclusion
# ----------
# In this demo we have seen how Quantum Phase Estimation works probably from a different perspective than you are used to. The great advantage of this approach we have shown you is that it creates a perfect bridge between classical signal processing and QPE. Therefore, future very important lines of research can be oriented to translate all the discoveries made in this work from the classical point of view into the quantum field.
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
