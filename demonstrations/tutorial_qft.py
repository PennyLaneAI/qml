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

    \text{QFT}|x_0 ...x_{n-1}\rangle = \bigotimes_{k = 0}^{n-1} \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ).$$

The nice thing about this representation is that we can see that there is no entanglement in the result and you could somehow prepare each qubit independently. Therefore, what we will do is to create these operators one by one. We will call them :math:`U_k` and they will be defined as follows:

.. math::
    U_k |x_0 ... x_{n-1}\rangle = |x0...x_{k-1}\rangle \otimes \left (|0\rangle + \exp \left (\frac{2\pi i 2^k}{2^n} x  \right) |1\rangle \right ) \otimes |x_{k+1}...x_{n-1}\rangle.

With such an operator we would simply have to apply it to each of the qubits:

..math::
    QFT = U_{n-1} ... U_1 U_0.

This operator can be easily built with controlled Phase Shift gates that add a phase only to :math:`|1\rangle`. We will represent these gates with a box in which we will include the phase they add. Below we show an animation in which this operator can be visualized for the particular case of :math:`n = 4` and :math:`k = 1.

.. figure:: ../_static/demonstration_assets/qft/qft.gif
    :align: center
    :width: 80%

As can be seen, each qubit adds a phase to the k-th qubit, proportional to its position in the binary representation. Note that qubits greater than $k$ need not contribute to the k-th qubit since they would make complete turns and it would not affect their value.
Therefore, the QFT for the case of 3 qubits would be:

.. figure::
    ../_static/demonstration_assets/qft/qft3.jpeg

"""


##############################################################################
# About the authors
# -----------------
#
