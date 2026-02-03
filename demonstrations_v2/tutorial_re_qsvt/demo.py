r"""
How to use PennyLane’s quantum resource estimator module for QSVT
=================================================================
.. meta::
    :property="og:description": Learn how to use PennyLane's estimator module to estimate the cost of QSVT
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/resource_estimation.jpeg

.. related::
    tutorial_intro_qsvt Intro to QSVT
    re_how_to_use_pennylane_for_resource_estimation How to use PennyLane for Resource Estimation
    tutorial_apply_qsvt QSVT in Practice

The Quantum Singular Value Transformation (QSVT) is a versatile algorithm, applicable to a wide range of
problems including unstructured search, Hamiltonian simulation, matrix inversion and many more [#chuang2021]_.
PennyLane makes it easy to build circuits and experiment with QSVT using the :func:`~.pennylane.qsvt` function.
For more information on how to use PennyLane's QSVT functionality checkout our other demos:

- `Intro to QSVT <tutorial_intro_qsvt>`_
- `QSVT in Practice <tutorial_apply_qsvt>`_
- `How to implement QSVT on hardware <tutorial_qsvt_hardware>`_

However, simulations can only take us so far, and industrially relevant system sizes are often too large to
meaningfully simulate. Fortunately, PennyLane's resource :mod:`~.pennylane.estimator` module can help us
gather meaningful insights in this regime. If you are new to resource estimation in PennyLane or need a quick
refresher, checkout this demo on `how to use PennyLane for Resource Estimation <re_how_to_use_pennylane_for_resource_estimation>`_.

In this demo, you will learn how to use PennyLane's :mod:`~.pennylane.estimator` module to estimate the cost of a
QSVT workflow.

Estimating the cost of QSVT
---------------------------
The logical cost of QSVT depends primarily on two factors, the **block encoding** operator and the **degree** of
the polynomial transformation. For example let's estimate the cost of performing a quintic (5th degree)
polynomial transformation to the matrix :math:`A`:

.. math::

    A = \begin{bmatrix}
        0.1 &  0.0 &  0.3 &  0.2 \\
        0.0 & -0.1 &  0.2 & -0.3 \\
        0.3 &  0.2 & -0.1 &  0.0 \\
        0.2 & -0.3 &  0.0 &  0.1 \\
        \end{bmatrix},


This particular matrix can be expressed as a linear combination of unitaries (LCU) :math:`A = 0.1 \cdot \hat{Z}_{0}\hat{Z}_{1} + 0.2 \cdot \hat{X}_{0}\hat{X}_{1} + 0.3 \cdot \hat{X}_{0}\hat{Z}_{1}`.
The LCU representation is crucial for building the **block encoding** operator using the standard method of LCUs.
For a recap on this technique, see our demo on `linear combination of unitaries and block encodings <tutorial_lcu_blockencoding>`_.
"""

import pennylane as qml

A = 0.1 * (qml.Z(0) @ qml.Z(1)) + 0.2 * (qml.X(0) @ qml.X(1)) + 0.3 * (qml.X(0) @ qml.Z(1))

print(qml.matrix(A, wire_order=[0, 1]))

##############################################################################
# Resources from an Executable Workflow
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Suppose we already had a PennyLane circuit which used QSVT to apply the quintic polynomial transformation to
# :math:`A`. Here we can use `estimate() <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.estimate.estimate.html>`_
# on the circuit directly to obtain the resource estimate.
#

import pennylane.numpy as qnp
import pennylane.estimator as qre

num_terms = len(A)
num_encoding_wires = int(qnp.ceil(qnp.log2(num_terms)))
encoding_wires = [f"e_{i}" for i in range(num_encoding_wires)]

poly = (0, 0, 0, 0, 0, 1)  # f(x) = x^5


def circ():
    qml.qsvt(A, poly, encoding_wires=encoding_wires)
    return


gs = {"X", "Y", "Z", "S", "T", "Hadamard", "CNOT", "Toffoli"}
resources = qre.estimate(circ, gate_set=gs)()
print(resources)

##############################################################################
# This works great for small systems and toy models. However, it suffers from performance bottlenecks when we
# scale to larger system sizes. In this case we can use some of the other functionality from the
# :mod:`~.pennylane.estimator` module to estimate the cost of QSVT.
#
# Resources from an Estimator Workflow
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First we use the :class:`~.pennylane.estimator.compact_hamiltonian.PauliHamiltonian` class to efficiently
# capture the LCU representation of :math:`A`. This produces a compact object specifically for resource
# estimation.
#
# First we build the block encoding operator by efficiently capturing the LCU representation of :math:`A` using
# the :class:`~.pennylane.estimator.compact_hamiltonian.PauliHamiltonian` class. This produces a compact object
# specifically for resource estimation. The block encoding operator is built with the
# `ChangeOpBasis <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.ops.ChangeOpBasis.html>`_
# class which uses the compute-uncompute pattern to implement the
# :math:`\text{Prep}^{\dagger} \circ \text{Select} \circ \text{Prep}` operator.
#
# The resources for `QSVT <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.QSVT.html>`_
# can then be obtained using this block encoding:

lcu_A = qre.PauliHamiltonian(
    num_qubits=2,
    pauli_terms={"ZZ": 1, "XX": 1, "XZ": 1},
)  # represents A = 0.1*ZZ + 0.2*XX + 0.3*XZ
num_terms = lcu_A.num_terms
num_qubits = lcu_A.num_qubits

num_encoding_wires = int(qnp.ceil(qnp.log2(num_terms)))
encoding_wires = [f"e_{i}" for i in range(num_encoding_wires)]
lcu_wires = [f"t_{i}" for i in range(num_qubits)]

Prep = qre.QubitUnitary(
    num_encoding_wires,
    wires=encoding_wires,
)  # Prep the coeffs of the LCU
Select = qre.SelectPauli(
    lcu_A,
    wires=lcu_wires + encoding_wires,
)  # Select over ops in the LCU
BlockEncoding = qre.ChangeOpBasis(Prep, Select)  # Prep ○ Sel Prep^t


qsvt_op = qre.QSVT(
    block_encoding=BlockEncoding,
    encoding_dims=(4, 4),  # The shape of matrix A
    poly_deg=5,  # quintic
)

resources = qre.estimate(qsvt_op, gate_set=gs)
print(resources)

##############################################################################
# Representing the QSVT workflow like this allows us to easily upscale it to larger system sizes without
# any computational overheads. Let's extend this example to a **50 qubit** system with an LCU of **2000
# terms** and a **100th degree** polynomial transformation. Notice how simple it is to update the code
# and obtain the cost of this larger system:

lcu_A = qre.PauliHamiltonian(
    num_qubits=50,
    pauli_terms={"ZZ": 250, "XX": 750, "XZ": 1000},
)  # 2000 terms !
num_terms = lcu_A.num_terms
num_qubits = lcu_A.num_qubits

num_encoding_wires = int(qnp.ceil(qnp.log2(num_terms)))
encoding_wires = [f"e_{i}" for i in range(num_encoding_wires)]
lcu_wires = [f"t_{i}" for i in range(num_qubits)]

Prep = qre.QROMStatePreparation(
    num_encoding_wires,
    wires=encoding_wires,
)  # More efficient Prep
Select = qre.SelectPauli(
    lcu_A,
    wires=lcu_wires + encoding_wires,
)  # Select over ops in the LCU
BlockEncoding = qre.ChangeOpBasis(Prep, Select)  #  Prep ○ Sel Prep^t

qsvt_op = qre.QSVT(
    block_encoding=BlockEncoding,
    encoding_dims=(2**50, 2**50),  # The shape of matrix A
    poly_deg=100,
)

resources = qre.estimate(qsvt_op, gate_set=gs)
print(resources)

##############################################################################
# With PennyLane's resource estimation functionality we can analyze the cost of QSVT workflows for large
# system sizes consisting of hundreds of qubits and millions of gates!
#
# Conclusion
# ----------
# In this demo, you learned how to use PennyLane's :mod:`~.pennylane.estimator` module to determine the
# resource requirements for **QSVT**. Now that you are armed with these tools for resource estimation,
# I challenge you to find another problem where polynomial transformations may be helpful, and figure out:
# what are the logical resource requirements of solving this on a quantum computer?
#
# References
# ----------
#
# .. [#chuang2021]
#
#     John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang,
#     "A Grand Unification of Quantum Algorithms"
#     `arxiv.2105.02859 <https://arxiv.org/abs/2105.02859>`__, 2021.
#
