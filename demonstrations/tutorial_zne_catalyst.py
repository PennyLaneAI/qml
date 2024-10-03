r"""
Digital Zero-Noise Extrapolation with Catalyst
==============================================

In this tutorial, you will learn how to use error mitigation, and in particular 
the Zero-Noise Extrapolation (ZNE) technique, in combination with `Catalyst <https://docs.pennylane.ai/projects/catalyst>`_, a framework for quantum
just-in-time (JIT) compilation with PennyLane. 
We'll demonstrate how to generate noise-scaled circuits, execute them on a noisy quantum
simulator, and use extrapolation techniques to estimate the zero-noise result, all while
leveraging JIT compilation through
Catalyst.

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
the demo :doc:`QJIT compilation with Qrack and Catalyst <qrack>`.

Stage 3: Combining the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After executing the noise-scaled circuits, an extrapolation on the results is performed  
to estimate the zero-noise limit---the result we would expect in a noise-free scenario. 
Catalyst provides two methods for perfoming this extrapolation:

1. **Polynomial extrapolation**, and
2. **Exponential extrapolation**.

Using ZNE with Pennylane
------------------------

The demo :doc:`Error mitigation with Mitiq and PennyLane <tutorial_error_mitigation>`
shows how ZNE, along with other error mitigation techniques, can be carried out in Pennylane by using `Mitiq <https://github.com/unitaryfund/mitiq>`__, 
a Python library developed by Unitary Fund.

ZNE in particular is also offered out of the box in Pennylane as a *differentiable* error mitigation technique,
for usage in combination with variational workflows. More on this in the tutorial 
:doc:`Differentiating quantum error mitigation transforms <tutorial_diffable-mitigation>`.

On top of the error mitigation routines offered in Pennylane, ZNE is also available for just-in-time 
(JIT) compilation, starting from Catalyst v0.8.1.
In this tutorial we see how an error mitigation routine can be integrated in a Catalyst workflow.

At the end of the tutorial, we will compare time for the execution of ZNE routines in 
pure Pennylane vs. Pennylane Catalyst with JIT. 

Defining the mirror circuit
---------------------------

The first step for demoing an error mitigation routine is to define a circuit. 
Here we build a simple mirror-circuit starting off a unitary 2-design. 
This is a typical construction for a randomized benchmarking circuit, which is used in many tasks
in quantum computing. Given such circuit, we measure the expectation value :math:`\langle Z\rangle` 
on the state of the first qubit, and by construction of the circuit, we expect this value to be
equal to 1.
"""

import os
import timeit

import pennylane as qml
import numpy as np
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
# As a sanity check, we first execute the circuit on the Qrack simulator without any noise.

noiseless_device = qml.device("qrack.simulator", n_wires, isNoisy=False, isOpenCL=False)

ideal_value = qml.QNode(circuit, device=noiseless_device)(w1, w2)
print(f"Ideal value: {ideal_value}")

##############################################################################
# As expected, in the noiseless scenario, the expecation value of the Pauli-Z measurement
# is equal to 1, since the first qubit is back in the :math:`|0\rangle` state.
#
# Mitigating the noisy circuit
# ----------------------------
# Let's now run the circuit through a noisy scenario. The Qrack simulator models noise by
# applying single-qubit depolarizing noise channels to all qubits in all gates of the circuit.
# The probability of error is specified by the value of the
# `QRACK_GATE_DEPOLARIZATION` environment variable.

NOISE_LEVEL = 0.01
os.environ["QRACK_GATE_DEPOLARIZATION"] = str(NOISE_LEVEL)
noisy_device = qml.device("qrack.simulator", n_wires, isNoisy=True, isOpenCL=False)

noisy_qnode = qml.QNode(circuit, device=noisy_device)
noisy_value = noisy_qnode(w1, w2)
print(f"Error without mitigation: {abs(ideal_value - noisy_value):.3f}")

##############################################################################
# Again expected, we obtain a noisy value that diverges from the ideal value we obtained above.
# Fortunately, we have error mitigation to the rescue! We can apply ZNE, however we are still
# missing some necessary parameters. In particular we still need to specify:
#
# 1. The method for scaling this noise up (in Catalyst there are two options: ``global`` and
#    ``local``)
# 2. The noise scaling factors (i.e. how much to increase the depth of the circuit)
# 3. The extrapolation technique used to estimate the ideal value (available in Catalyst are
#    polynomial and exponential extrapolation).
#
# First, we choose a method to scale the noise. This needs to be specified as a Python string.

folding_method = "global"

##############################################################################
# Next, we pick a list of scale factors. At the time of writing this tutorial,
# Catalyst supports only odd integer scale factors. In the global folding setting,
# a scale factor  :math:`s` correspond to the circuit being folded
# :math:`\frac{s - 1}{2}` times.
scale_factors = [1, 3, 5]

##############################################################################
# Finally, we'll choose the extrapolation technique. Both exponential and polynomial extrapolation
# is available in the :mod:`qml.transforms <pennylane.transforms>` module, and both of these functions can be passed directly
# into Catalyst's :func:`catalyst.mitigate_with_zne` function. In this tutorial we use polynomial extrapolation,
# which we hypothesize it best models the behavior of the noise scenario we are considering.

from pennylane.transforms import poly_extrapolate
from functools import partial

extrapolation_method = partial(poly_extrapolate, order=3)

##############################################################################
# We're now ready to run our example using ZNE with Catalyst! Putting these all together we're able
# to define a very simple :func:`~.QNode`, which represents the mitigated version of the original circuit.


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
# It's crucial to note that we can use the :func:`~.qjit` decorator here, as all the functions used
# to define the node are compatible with Catalyst, and we can therefore
# exploit the potential of just-in-time compilation.
#
# Benchmarking
# ------------
# For comparison, let's define a very similar :func:`~.qnode`, but this time we don't decorate the node
# as just-in-time compilable.
# When it comes to the parameters, the only difference here (due to an implementation technicality)
# is the type of the ``folding`` argument. Despite the type being different, however,
# the value of the folding method is the same, i.e., global folding.


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
# To showcase the impact of JIT compilation, let's use Python's ``timeit`` module
# to measure execution time of ``mitigated_circuit_qjit`` vs. ``mitigated_circuit``:

repeat = 5  # number of timing runs
number = 5  # number of loops executed in each timing run

times = timeit.repeat("mitigated_circuit(w1, w2)", globals=globals(), number=number, repeat=repeat)

print(f"mitigated_circuit running time (best of {repeat}): {min(times):.3f}s")

times = timeit.repeat(
    "mitigated_circuit_qjit(w1, w2)", globals=globals(), number=number, repeat=repeat
)

print(f"mitigated_circuit_qjit running time (best of {repeat}): {min(times):.3f}s")

##############################################################################
# Already with the simple circuit we started with, and with the simple parameters in our example,
# we can appreciate the performance differences. That was at the cost of very minimal syntax change.
#
# There are still reasons to use ZNE in Pennylane without :func:`~.qjit`, for instance,
# whenever the device of choice is not supported by Catalyst. To help,
# we conlcude with a landscape of the QEM techniques available on Pennylane.
#
# .. list-table::
#     :widths: 30 20 20 20 20 30
#     :header-rows: 1
#
#     * - **Framework**
#       - **ZNE folding**
#       - **ZNE extrapolation**
#       - **Differentiable**
#       - **JIT**
#       - **other QEM techniques**
#     * - Pennylane + Mitiq
#       - global, local, random
#       - polynomial, exponential
#       -
#       -
#       - PEC, CDR, DDD, REM
#     * - Pennylane transforms
#       - global, local
#       - polynomial, exponential
#       - ✅
#       -
#       -
#     * - Catalyst (experimental)
#       - global, local
#       - polynomial, exponential
#       - ✅
#       - ✅
#       -


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
