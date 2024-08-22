r"""
Digital Zero-Noise Extrapolation with Catalyst
========================================

In this tutorial you will learn how to use ... and quantum just-in-time (QJIT) compilation via
`Catalyst
<https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.

What is ZNE
---------------

"""
import os

import jax

import pennylane as qml
from pennylane import numpy as np
from catalyst import qjit, mitigate_with_zne


#############################################################################
# We use Qrack because it implements a noise-model and it's compatible with Catalyst
qubits = 2
NOISE_LEVEL = 0.1
os.environ["QRACK_GATE_DEPOLARIZATION"] = str(NOISE_LEVEL)
dev = qml.device("qrack.simulator", qubits, isNoisy=True)


@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(wires=0))


folding_factors = jax.numpy.array([1, 2, 3])

@qjit
def mitigated_circuit():
    return mitigate_with_zne(
        circuit, 
        scale_factors=folding_factors
    )()

print(circuit())
print(mitigated_circuit())


##############################################################################
# 
# References
# ----------
#
# .. [#DZNEpaper]
#
#     Tudor Giurgica-Tiron, Yousef Hindy, Ryan LaRose, Andrea Mari, and William J. Zeng
#     "Digital zero noise extrapolation for quantum error mitigation"
#     `arXiv:2005.10921v2 <"https://arxiv.org/abs/2005.10921v2">`__, 2020.
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/nate_stemen.txt
