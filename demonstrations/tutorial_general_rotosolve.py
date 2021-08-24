r"""

.. _general_rotosolve:

Rotosolve for general quantum gates
===================================

.. meta::
    :property="og:description": Use the Rotosolve optimizer for general quantum gates.
    :property="og:image": general_rotosolve/general_rotosolve_thumbnail.png

.. related::

   tutorial_rotoselect Leveraging trigonometry with Rotoselect
   tutorial_general_parshift Parameter-shift rules for general quantum gates
   tutorial_expressivity_fourier_series Understand quantum models as Fourier series


*Author: David Wierichs (Xanadu resident). Posted: ?? September 2021.*

.............................

|

.. figure:: general_rotosolve/general_rotosolve_thumbnail.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Rotosolve optimization for higher-order Fourier series cost functions.


In this demo we will show how to use the :class:`~.pennylane.optimize.RotosolveOptimizer`
on more complicated quantum circuits than those with individually parametrized Pauli rotations.



VQEs give rise to trigonometric cost functions
----------------------------------------------
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)

# Create a device with 2 qubits.
dev = qml.device("default.qubit", wires=2)

# Define the variational form V and observable M and combine them into a QNode.
@qml.qnode(dev, diff_method="parameter-shift")
def circuit(parameters):
    qml.RX(parameters[0], wires=0)
    qml.RX(parameters[1], wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


###############################################################################
# blabla
#
# References
# ----------
#
# .. [#QAD]
#
#     Balint Koczor, Simon Benjamin. "Quantum Analytic Descent".
#     `arXiv preprint arXiv:2008.13774 <https://arxiv.org/abs/2008.13774>`__.
#
# .. [#Rotosolve]
#
#     Mateusz Ostaszewski, Edward Grant, Marcello Benedetti.
#     "Structure optimization for parameterized quantum circuits".
#     `arXiv preprint arXiv:1905.09692 <https://arxiv.org/abs/1905.09692>`__.
#
# .. [#higher_order_diff]
#
#     Andrea Mari, Thomas R. Bromley, Nathan Killoran.
#     "Estimating the gradient and higher-order derivatives on quantum hardware".
#     `Phys. Rev. A 103, 012405 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.012405>`__, 2021.
#     `arXiv preprint arXiv:2008.06517 <https://arxiv.org/abs/2008.06517>`__.
