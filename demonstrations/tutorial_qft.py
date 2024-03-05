r"""Intro to Quantum Fourier Transform
=============================================================

intro QFT

The Quantum Fourier Transform (QFT)
-----------------------------------

FT and QFT

Building the Quantum Fourier Transform
--------------------------------------

Although we have already shown a formula for the QFT above, there is another equivalent representation that will be particularly useful:

.. math::

    \text{QFT}|x\rangle = \bigotimes_{k = n-1}^{0} \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ).

The nice thing about this representation is that it shows that there is no entanglement between qubits and we could somehow prepare each qubit independently. We will call :math:`U_k` the operator that is able to prepare the k-th qubit. This operator is defined as:

.. math::

    U_k |x\rangle = |x_0 \dots x_{k-1}\rangle \otimes \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ) \otimes |x_{k+1} \dots x_{n-1}\rangle,

where :math:`|x\rangle = |x0 \dots x_{n-1}\rangle`.
We can build :math:`U_k` with one Hadamard gate and controlled Phase Shift gates, which add a phase only to :math:`|1\rangle`. We will represent these control gates with a box in which we will include inside it the phase they add. Below we show an animation in which this operator can be visualized for the particular case of :math:`n = 4` and :math:`k = 1`.

.. figure:: ../_static/demonstration_assets/qft/qft.gif
    :align: center
    :width: 80%

Each qubit adds a phase to the k-th qubit, proportional to its position in the binary representation. Note that qubits over the :math:`k`-th one don't contribute to the k-th qubit since they would make complete turns and it would not affect their value.
With this operator we can now prepare the QFT by applying them sequentially:

.. math::

    QFT = U_{n-1} \dots U_1 U_0.


Therefore, the QFT for the case of 3 qubits would be:

.. figure::
    ../_static/demonstration_assets/qft/qft3.jpeg

One last detail to consider, is that in the QFT formula we have given, the values of the k's goes in reversed order from :math:`n-1` to :math:`0`.
That is why we should make a last step and change the order of all the qubits. For this reason, it is common to find some swap operators at the end of the template.
"""

import pennylane as qml
from functools import partial
import matplotlib.pyplot as plt

# This line is to expand the circuit to see the operators
@partial(qml.devices.preprocess.decompose, stopping_condition = lambda obj: False, max_expansion=1)

def circuit():
  qml.QFT(wires = range(4))

qml.draw_mpl(circuit, decimals = 2, style = "pennylane")()
plt.show()


##############################################################################
# About the authors
# -----------------
#
