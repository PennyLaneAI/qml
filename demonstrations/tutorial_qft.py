r"""Intro to Quantum Fourier Transform
=============================================================

If you have found this tutorial it is because you want to learn what the quantum fourier transform (QFT) is. You've probably heard of it is one
of the most popular algorithms which we can find it in places like Quantum Phase Estimation or even in the well known
Shor's algorithm.

The QFT is the quantum analog to the discrete Fourier transform, or DFT, the main tool of digital signal processing.
These methods are defined as a transformation that allows to change the time domain into frequency domain. If you are
not familiar with that, what this transformation achieves is to facilitate the manipulation and understanding of periodic functions.

In this tutorial you will learn how to define this operation and learn how to build it with basic gates.

Defining the Quantum Fourier Transform
---------------------------------------

The discrete Fourier transform takes as input a vector :math:`(x_0, \dots, x_{N-1})` and returns another vector :math:`(y_0, \dots, y_{N-1})` where:

.. math::
  y_k = \frac{1}{\sqrt{N}} \sum_{j = 0}^{N-1} x_j e^{-\frac{2\pi i kj}{N}}.

For ease of comparison we will assume :math:`N = 2^n`. The idea of the QFT is to perform the same operation but in a quantum state :math:`|x\rangle = \sum_{i = 0}^{N-1} x_i |i\rangle`.
In this case, the output will be another quantum state :math:`|y\rangle = \sum_{i = 0}^{N-1} y_i |i\rangle` where:

.. math::
    y_k = \frac{1}{\sqrt{N}} \sum_{j = 0}^{N-1} x_j e^{\frac{2\pi i kj}{N}}.

For historical reasons, there is a change of notation and the sign of the exponent is different. It is for this reason
that the DFT coincides with :math:`QFT^{\dagger}` instead of the QFT.

These transformations are linear and can be represented by a matrix. Let's see that the matrices actually match!
"""

from scipy.linalg import dft
import pennylane as qml
import numpy as np


n = 2

print("DFT matrix for n = 2:\n")
print(np.round(1 / np.sqrt(2 ** n) * dft(2 ** n), 2))

qft_inverse = qml.adjoint(qml.QFT([0,1]))

print("\n inverse QFT matrix for n = 2:\n")
print(np.round(qft_inverse.matrix(), 2))

#############################################
# Great, the generated matrices are the same.
# An important factor to consider however is the algorithmic complexity of QFT. While the classical version has a complexity
# :math:`\mathcal{O}(n2^n)`, QFT only needs to apply :math:`\mathcal{O}(n^2)` operations.
#
# Building the Quantum Fourier Transform
# --------------------------------------
#
# Although we have already shown a formula for the QFT above, there is another equivalent representation that will be particularly useful:
#
# .. math::
#
#    \text{QFT}|x\rangle = \bigotimes_{k = n-1}^{0} \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ).
#
# The nice thing about this representation is that it gives us an independent expression for each qubit and we could somehow prepare them independently. We will call :math:`U_k` the operator that is able to prepare the k-th qubit. This operator is defined as:
#
# .. math::
#
#    U_k |x\rangle = |x_0 \dots x_{k-1}\rangle \otimes \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ) \otimes |x_{k+1} \dots x_{n-1}\rangle,
#
# where :math:`|x\rangle = |x_0 \dots x_{n-1}\rangle`.
# We can build :math:`U_k` with one Hadamard gate and controlled Phase Shift gates, which add a phase only to :math:`|1\rangle`. Below we show an animation in which this operator can be visualized for the particular case of :math:`n = 4` and :math:`k = 1`.
# We will represent these control gates with a box in which we will include inside it the phase they add.
#
#.. figure:: ../_static/demonstration_assets/qft/qft.gif
#    :align: center
#    :width: 80%
#
#
# Each qubit adds a phase to the k-th qubit, proportional to its position in the binary representation. Note that qubits over the :math:`k`-th one don't contribute to the phase since they would make complete turns and it would not affect its value.
# With this operator we can now prepare the QFT by applying them sequentially:
#
# .. math::
#
#    QFT = U_{n-1} \dots U_1 U_0.
#
#
# Therefore, the QFT for the case of 3 qubits would be:
#
# .. figure::
#   ../_static/demonstration_assets/qft/qft3.jpeg
#
# One last detail to consider, is that in the QFT formula we have given, the index :math:`k` goes in reversed order from :math:`n-1` to :math:`0`.
# That is why we should make a last step and change the order of all the qubits. For this reason, it is common to find some swap operators at the end of the template.
#
# Example
# -------
#
# Code example
#
# About the authors
# -----------------
#