r""".. _mbqc:

Measurement-based quantum computation
=============================

.. meta::
    :property="og:description": Learn about measurement-based quantum computation
    :property="og:image": https://pennylane.ai/qml/_images/mbqc.png

*Author: Radoica Draskic & Joost Bus. Posted: Day Month 2022. Last updated: 4 May 2022.*

**Measurement-based quantum computation** [#OneWay2021] is one of the prososals of a physical implementation of a quantum Turing machine.

"""
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


##############################################################################
#
# Cluster states
# --------------
#
# .. figure:: ../demonstrations/mbqc/mbqc_blueprint.png
#    :align: center
#    :width: 60%
#
#    ..
#
#    Cluster state proposed [#XanaduBlueprint2021]
#
# To understand how MBQC qubits work, we first need to explain what cluster states are...


##############################################################################
# References
# ----------
#
#
# .. [#OneWay2021]
#
#     Robert Raussendorf and Hans J. Briegel (2021) "A One-Way Quantum Computer",
#     `Phys. Rev. Lett. 86, 5188
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188>`__.
#
# .. [#XanaduBlueprint2021]
#
#     J. Eli Bourassa, Rafael N. Alexander, Michael Vasmer et al. (2021) "Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer",
#     `Quantum 5, 392
#     <https://quantum-journal.org/papers/q-2021-02-04-392/>`__.
#
