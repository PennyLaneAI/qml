r"""
Digital Zero-Noise Extrapolation with Catalyst
==============================================

In this tutorial, you will learn how to use Zero-Noise Extrapolation (ZNE) in combination with
Catalyst. We'll demonstrate how to generate noise-scaled circuits, execute them on a noisy quantum
simulator, and use extrapolation techniques to estimate the zero-noise result, all while
leveraging just-in-time (JIT) compilation through
`Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_.

What is ZNE
-----------

Zero-Noise Extrapolation (ZNE) is a technique used to mitigate the effect of noise on quantum
computations. First introduced in [#temme2017zne]_, it helps improve the accuracy of quantum
results by running circuits at varying noise levels and extrapolating back to a hypothetical
zero-noise case. While this tutorial won't delve into the theory behind ZNE in detail, lets first
review what happens when using the protocol in practice.

Stage 1: Generating Noise-Scaled Circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ZNE to work, we need to generate circuits with **increased** noise. Currently, ZNE in Catalyst
supports two methods for generating noise-scaled circuits:

1. **Global folding**: If a circuit implements a global unitary :math:`U`, global folding applies
   :math:`U(U^\dagger U)^n` for some integer :math:`n`,
   effectively scaling the noise in the entire circuit.
2. **Local folding**: Individual gates are repeated (or folded) in contrast with the entire
   circuit.

Stage 2: Running the circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once noise-scaled circuits are created, they need to be run! These can be executed on either real
quantum hardware or a noisy quantum simulator. In this tutorial, we'll use the
`Qrack quantum simulator <https://qrack.readthedocs.io/>`_, which implements a noise model
compatible with Catalyst.

Stage 3: Combining the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After executing the noise-scaled circuits, we perform an extrapolation to estimate the zero-noise
limit---the result we would expect in a noise-free scenario. Catalyst provides two methods for
perfoming this extrapolation:

1. **Polynomial extrapolation**, and
2. **Exponential extrapolation**.


Using ZNE in Catalyst
---------------------

Let's see what this workflow looks like using Catalyst.
To get things started we'll need two ingredients:

1. an example quantum circuit to run
2. a way to execute circuits

The circuits we use here will be ..., and the method of execution will be a noisy simulation
available through Qrack. The underlying noise model of this simulation is that of depolarizing
noise.

"""

import os

import jax

import pennylane as qml
from pennylane import numpy as np
from catalyst import qjit, mitigate_with_zne


qubits = 2
NOISE_LEVEL = 0.1
os.environ["QRACK_GATE_DEPOLARIZATION"] = str(NOISE_LEVEL)
dev = qml.device("qrack.simulator", qubits, isNoisy=True)


@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(wires=0))


##############################################################################
# With a circuit and simulator defined, we can begin to define some of the necessary parameters
# In particular we will need to specify:
#
# 1. The noise scaling factors (i.e. how much to increase the depth of the circuit)
# 2. The _method_ for scaling this noise up (in Catalyst there are two options: `global` and
#    `local`)
# 3. The extrapolation technique to use to estimate the ideal value (available in Catalyst are
#    polynomial, and exponential extrapolation).
#
# We'll define the scale factors as a `jax.numpy.array` where the scale factors must be odd
# integers.

scale_factors = jax.numpy.array([1, 3, 5])

##############################################################################
# Next, we'll choose a method to scale the noise. This needs to be defined as a Python string.

folding_method = "local"

##############################################################################
# Finally, we'll choose the extrapolation technique. Both exponential and polynoamial extrapolation
# is available in the `pennylane.transforms` module. Both of these functions can be passed directly
# into `mitigate_with_zne`!

from pennylane.transforms import exponential_extrapolate, poly_extrapolate

extrapolation_method = exponential_extrapolate

##############################################################################
# We're now ready to run our example using ZNE with Catalyst! Putting these all together we're able
# to define a very simple `QNode`!


@qjit
def mitigated_circuit():
    return mitigate_with_zne(
        circuit,
        scale_factors=scale_factors,
        extrapolate=extrapolation_method,
        folding=folding_method,
    )()


print(circuit())
print(mitigated_circuit())

##############################################################################
# But there's still a big unanswered question! _If I can do this all in PennyLane, what is Catalyst
# offering here?_ That's a **great** question! In order to explore the difference we'll need to
# explore what happens when `catalyst.qjit` is, and is not, used!
# ...


##############################################################################
#
# References
# ----------
#
# .. [#temme2017zne] K. Temme, S. Bravyi, J. M. Gambetta
#     `"Error Mitigation for Short-Depth Quantum Circuits" <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`_,
#     Phys. Rev. Lett. 119, 180509 (2017).
#
# .. [#DZNEpaper]
#     Tudor Giurgica-Tiron, Yousef Hindy, Ryan LaRose, Andrea Mari, and William J. Zeng
#     "Digital zero noise extrapolation for quantum error mitigation"
#     `arXiv:2005.10921v2 <https://arxiv.org/abs/2005.10921v2>`__, 2020.
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/nate_stemen.txt
