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
zero-noise case. While this tutorial won't delve into the theory behind ZNE in detail, let's first
review what happens when using the protocol in practice.

Stage 1: Generating Noise-Scaled Circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In its digital version [#DZNEpaper]_, ZNE works by generating circuits with **increased** noise. 
Currently, ZNE in Catalyst supports two methods for generating noise-scaled circuits:

1. **Global folding**: If a circuit implements a global unitary :math:`U`, global folding applies
   :math:`U(U^\dagger U)^n` for some integer :math:`n`,
   effectively scaling the noise in the entire circuit.
2. **Local folding**: Individual gates are repeated (or folded) in contrast with the entire
   circuit.

Stage 2: Running the circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once noise-scaled circuits are created, they need to be run! These can be executed on either real
quantum hardware or a noisy quantum simulator. In this tutorial, we'll use the
`Qrack quantum simulator <https://qrack.readthedocs.io/>`_, which is both compatible with Catalyst,
and implements a noise model. For more about the integration of Qrack and Catalyst, see
the demo `QJIT compilation with Qrack and Catalyst <https://pennylane.ai/qml/demos/qrack/>`_.

Stage 3: Combining the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After executing the noise-scaled circuits, we perform an extrapolation to estimate the zero-noise
limit---the result we would expect in a noise-free scenario. Catalyst provides two methods for
perfoming this extrapolation:

1. **Polynomial extrapolation**, and
2. **Exponential extrapolation**.

Using ZNE with Pennylane
------------------------
The demo `Error mitigation with Mitiq and PennyLane <https://pennylane.ai/qml/demos/tutorial_error_mitigation/>`_
shows how ZNE, along with other error mitigation techniques, can be carried out in Pennylane by using Mitiq, 
a Python library developed by Unitary Fund.

ZNE in particular is also offered out of the box in Pennylane as a *differentiable* error mitigation technique,
for usage in combination with variational workflows. More on this in the tutorial 
`Differentiating quantum error mitigation transforms <https://pennylane.ai/qml/demos/tutorial_diffable-mitigation/>`_.


Using ZNE in Catalyst
---------------------

ZNE is also available for just-in-time (JIT) compilation of PennyLane programs, 
starting from Catalyst v0.8.1.
In this tutorial we see how an error mitigation routine can be integrated in a Catalyst workflow.

At the end of the tutorial, we will compare time for the execution of ZNE routines in 
Pennylane vs. Catalyst.

Mirror circuit
--------------

First we build a mirror-circuit starting off a unitary 2-design. 
This is a typical construction for a randomized benchmarking circuit, which is used in many tasks
in quantum computing. Given such circuit, we measure the expectation value :math:`\langle Z\rangle` 
on the state of the first qubit.
"""

import os
import timeit

import pennylane as qml
from pennylane import numpy as np
from catalyst import qjit, mitigate_with_zne

n_wires = 5

np.random.seed(42)

n_layers = 10
template = qml.SimplifiedTwoDesign
weights_shape = template.shape(n_layers, n_wires)
w1, w2 = [2 * np.pi * np.random.random(s) for s in weights_shape]


def circuit(w1, w2):
    template(w1, w2, wires=range(n_wires))
    qml.adjoint(template)(w1, w2, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

##############################################################################
# We execute the circuit on the Qrack simulator, first without noise, 
# and then in a noisy scenario.

noiseless_device = qml.device("qrack.simulator", n_wires, isNoisy=False)

ideal_value = qml.QNode(circuit, device=noiseless_device)(w1, w2)
print(f"Ideal value: {ideal_value}")

##############################################################################
# As expected, in the noiseless scenario, the first qubit is in the :math:|0\rangle`
# state, so the expecation value of the Pauli-Z measurement is equal to 1.

NOISE_LEVEL = 0.01
os.environ["QRACK_GATE_DEPOLARIZATION"] = str(NOISE_LEVEL)
noisy_device = qml.device("qrack.simulator", n_wires, isNoisy=True)

noisy_qnode = qml.QNode(circuit, device=noisy_device)
noisy_value = noisy_qnode(w1, w2)
print(f"Error without mitigation: {abs(ideal_value - noisy_value):.8f}")

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
# We'll define the scale factors as a `jax.numpy.array` where the scale factors represents
# the number of time the circuit is folded.

scale_factors = [1, 3, 5]

##############################################################################
# Next, we'll choose a method to scale the noise. This needs to be defined as a Python string.

folding_method = "global"

##############################################################################
# Finally, we'll choose the extrapolation technique. Both exponential and polynoamial extrapolation
# is available in the `pennylane.transforms` module. Both of these functions can be passed directly
# into `mitigate_with_zne`!

from pennylane.transforms import richardson_extrapolate

extrapolation_method = richardson_extrapolate

##############################################################################
# We're now ready to run our example using ZNE with Catalyst! Putting these all together we're able
# to define a very simple `QNode`!


@qjit
def mitigated_circuit_qjit(w1, w2):
    return mitigate_with_zne(
        noisy_qnode,
        scale_factors=scale_factors,
        extrapolate=extrapolation_method,
        folding=folding_method,
    )(w1, w2)


zne_value = mitigated_circuit_qjit(w1, w2)

print(f"Error with ZNE in Catalyst: {abs(ideal_value - zne_value):.3f}")


##############################################################################
# ZNE in Pennylane
# --------------------------------------------------------
# Notice

def mitigated_circuit(w1, w2):
    return qml.transforms.mitigate_with_zne(
        noisy_qnode,
        scale_factors=scale_factors,
        extrapolate=extrapolation_method,
        folding=qml.transforms.fold_global,
    )(w1, w2)

zne_value = mitigated_circuit(w1, w2)

print(f"Error with ZNE in Pennylane: {abs(ideal_value - zne_value):.3f}")

##############################################################################
# We use the timeit module in Python to measure execution time of a mitigated circuit
# ...

repeat = 5  # number of timing runs
number = 5  # number of loops executed in each timing run 

times = timeit.repeat(
    "mitigated_circuit(w1, w2)", 
    globals=globals(), number=number, repeat=repeat)

print(f"Pennylane running time (best of {repeat}): {min(times):.3f}")

times = timeit.repeat(
    "mitigated_circuit_qjit(w1, w2)", 
    globals=globals(), number=number, repeat=repeat)

print(f"Catalyst running time (best of {repeat}): {min(times):.3f}")




##############################################################################
# Here is a recap of the landscape of QEM techniques available in Pennylane.
#
#     .. list-table::
#        :widths: 30 35 20 15
#        :header-rows: 1
#
#        * - **Framework**
#          - **Techniques**
#          - **Differentiable**
#          - **JIT**
#        * - Pennylane + Mitiq
#          - ZNE, PEC, CDR, DDD, REM
#          - 
#          - 
#        * - Pennylane transforms
#          - ZNE
#          - ✅
#          - 
#        * - Catalyst
#          - ZNE
#          - ✅
#          - ✅


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
