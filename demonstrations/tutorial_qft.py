r"""Intro to Quantum Fourier Transform
=============================================================

The quantum Fourier transform (QFT) is one of the most important building blocks in quantum algorithms, famously used in [quantum phase estimation](https://pennylane.ai/qml/demos/tutorial_qpe/) and [Shor's factoring algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm).

The QFT is the quantum analog to the discrete Fourier transform, or DFT, the main tool of digital signal processing.
These methods are defined as a transformation that allows to change the time domain into frequency domain. If you are
not familiar with that, what this transformation achieves is to facilitate the manipulation and understanding of periodic functions.

In this tutorial you will learn how to define this operation and how to build it with basic gates.

Defining the Quantum Fourier Transform
---------------------------------------

To appreciate the definition of QFT, it will help to start with its classical counterpart.
The discrete Fourier transform takes as input a vector :math:`(x_0, \dots, x_{N-1}) \in \mathbb{C}^N` and returns another vector :math:`(y_0, \dots, y_{N-1}) \in \mathbb{C}^N` where:

.. math::
  y_k = \sum_{j = 0}^{N-1} x_j e^{-\frac{2\pi i kj}{N}}.

For ease of comparison we will assume :math:`N = 2^n`. The idea of the QFT is to perform the same operation but in a quantum state :math:`|x\rangle = \sum_{i = 0}^{N-1} x_i |i\rangle`.
In this case, the output will be another quantum state :math:`|y\rangle = \sum_{i = 0}^{N-1} y_i |i\rangle` where:

.. math::
    y_k = \frac{1}{\sqrt{N}} \sum_{j = 0}^{N-1} x_j e^{\frac{2\pi i kj}{N}}.

For historical reasons, there is a change of notation and the sign of the exponent is positive. It is for this reason
that the DFT coincides with :math:`\text{QFT}^{\dagger}` instead of the QFT. Also, in the QFT we include normalization factor :math:`\frac{1}{\sqrt{N}}`.

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
# Also it is important to consider the algorithmic complexity of the QFT. While the classical version has a complexity :math:`\mathcal{O}(n2^n)`, QFT only needs to apply :math:`\mathcal{O}(n^2)` operations.
# This is a huge advantage when we are working with large quantum systems.
#
# Building the Quantum Fourier Transform
# --------------------------------------
#
# Although we have already shown a formula for the QFT above, there is another equivalent representation that will be particularly useful:
#
# .. math::
#
#    \text{QFT}|x\rangle = \bigotimes_{k = n-1}^{0} \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ),
#
# for :math:`x \in [0, \dots, N-1]`. The nice thing about this representation is that it gives us an independent expression for each qubit and we could somehow prepare them independently. We will call :math:`U_k` the operator that is able to prepare the k-th qubit. This operator is defined as:
#
# .. math::
#
#    U_k |x\rangle = |x_0 \dots x_{k-1}\rangle \otimes \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ) \otimes |x_{k+1} \dots x_{n-1}\rangle,
#
# where :math:`|x\rangle = |x_0 \dots x_{n-1}\rangle`.
# We can build :math:`U_k` with one Hadamard gate and controlled Phase Shift gates, which add a phase only to :math:`|1\rangle`. Below we show an animation in which this operator can be visualized for the particular case of :math:`n = 4` and :math:`k = 1`.
# We will represent these control gates with a box in which we will include inside it the phase they add.
#
# .. figure:: ../_static/demonstration_assets/qft/qft_gif.gif
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

import pennylane as qml
from functools import partial
import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

# This line is to expand the circuit to see the operators
@partial(qml.devices.preprocess.decompose, stopping_condition = lambda obj: False, max_expansion=1)

def circuit():
  qml.QFT(wires = range(4))

qml.draw_mpl(circuit, decimals = 2, style = "pennylane")()
plt.show()

#############################################
# Using the QFT
# --------------
#
# So far so good. We have seen how to define it, and we have shown how to build it with basic gates.
# Now it's time to put it into practice. Let's imagine that we have a ``prep`` operator, which prepare some state:


def prep():
    """quntum function that prepares the state."""

    qml.PauliX(wires=0)
    for wire in range(1,6):
        qml.Hadamard(wires=wire)
    qml.ControlledSequence(qml.PhaseShift(-2 * np.pi / 10, wires=0), control=range(1,6))
    qml.PauliX(wires=0)


dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():

    prep()

    return qml.state()


state = circuit().real[:32]

plt.bar(range(len(state)), state)
plt.xlabel("|x⟩")
plt.ylabel("Amplitude (real part)")
plt.show()

#############################################
# This 5-qubit state is showing a periodic behavior and we would like to know its period. A first approach could be
# to measure the state and try to understand its shape but this would not be efficient. That is why we will use the QFT,
# which is able to change the state into frequency domain.
#

@qml.qnode(dev)
def circuit():

  prep()
  qml.QFT(wires = range(1,6))

  return qml.probs(wires = range(1,6))

state = circuit()[:32]

plt.bar(range(len(state)), state)
plt.xlabel("|x⟩")
plt.ylabel("probs")
plt.show()

#############################################
# Now we can see that the state has changed and it is showing a clear peak at the position :math:`|x\rangle = 3`
# This value corresponds to an approximation of :math:`2^nf` where :math:`f` is the frequency and :math:`n` the
# number of qubits.
# By using the formula:
#
# .. math::
#
#    T = \frac{1}{f},
#
# we can find the period :math:`T` of our state.
# In this case, the period is :math:`T = 2^5 / 3 \sim 10.33` close to the real value of :math:`10`.
# The preparation of this state, is a real example of Quantum Phase Estimation. A first block prepares a state where a
# certain value is encoded in the period and then, we use the QFT to find that value.
#
#
# Conclusion
# ----------
# In this tutorial, we've journeyed through the fundamentals and construction of the Quantum Fourier Transform (QFT), a
# cornerstone in quantum computing analogous to the classical Discrete Fourier Transform. We explored its mathematical
# formulation, its implementation with basic quantum gates, and its application demonstrating its powerful utility in algorithms like Quantum Phase Estimation.
# It is a technique that deserves to be mastered to detect potential key applications in the future!
#
# About the authors
# -----------------
#