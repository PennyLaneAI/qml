r"""Intro to Quantum Phase Estimation
=============================================================

intro QFT

The Quantum Fourier Transform (QFT)
-----------------------------------

FT and QFT

Building the Quantum Fourier Transform
--------------------------------------

Although we have already shown a formula for the QFT above, there is another equivalent representation that will be particularly useful:

.. math::

    \text{QFT}|x\rangle = \bigotimes_{k = 0}^{n-1} \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ).

The nice thing about this representation is that it shows that there is no entanglement between qubits and we could somehow prepare each qubit independently. We will call :math:`U_k` the operator that is able to prepare the k-th qubit. This operator is defined as:

.. math::

    U_k |x\rangle = |x0 \dots x_{k-1}\rangle \otimes \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ) \otimes |x_{k+1} \dots x_{n-1}\rangle,

where :math:`|x\rangle = |x0 \dots x_{k-1}\rangle \otimes |x_k\rangle \otimes |x_{k+1} \dots x_{n-1}\rangle`.
We can build :math:`U_k` with controlled Phase Shift gates that add a phase only to :math:`|1\rangle`. We will represent these gates with a box in which we will include the phase they add. Below we show an animation in which this operator can be visualized for the particular case of :math:`n = 4` and :math:`k = 1`.

.. figure:: ../_static/demonstration_assets/qft/qft.gif
    :align: center
    :width: 80%

As can be seen, each qubit adds a phase to the k-th qubit, proportional to its position in the binary representation. Note that qubits greater than :math:`k` need not contribute to the k-th qubit since they would make complete turns and it would not affect their value.
With this operator we can now prepare the QFT by applying them sequentially:

.. math::

    QFT = U_{n-1} \dots U_1 U_0.


Therefore, the QFT for the case of 3 qubits would be:

.. figure::
    ../_static/demonstration_assets/qft/qft3.jpeg


"""


##############################################################################
# About the authors
# -----------------
#
