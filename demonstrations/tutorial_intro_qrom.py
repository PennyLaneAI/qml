r"""Intro to quantum read-only memory (QROM)
=============================================================

Managing data is a crucial task, and quantum computers are no exception: efficient data management is vital in `quantum machine learning <https://pennylane.ai/qml/quantum-machine-learning/>`__, search algorithms, and :doc:`state preparation </demos/tutorial_initial_state_preparation/>`.
In this demonstration, we will discuss the concept of a quantum read-only memory (QROM), a data structure designed to load classical data into a quantum computer.
This is a valuable tool in quantum machine learning or for preparing quantum states among others.
We also explain how to use this operator in PennyLane using the :class:`~.pennylane.QROM` template.


.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_qrom.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

QROM
-----

Quantum read-only memory (QROM) is an operator that allows us to load classical data into a quantum computer. Data are represented as a collection of bitstrings (lists composed of 0s and 1s) that we denote by :math:`b_0, b_1, \ldots, b_{N-1}`. The QROM operator is then defined as:

.. math::

    \text{QROM}|i\rangle|0^{\otimes m}\rangle = |i\rangle|b_i\rangle,

where :math:`|b_i\rangle` is the bitstring associated with the :math:`i`-th computational basis state, and :math:`m` is the length of the bitstrings. We have assumed all the bitstrings are of equal length.

For example, suppose our data consists of eight bitstrings, each with two bits: :math:`[01, 11, 11, 00, 01, 11, 11, 00]`. Then, the index register will consist of three
qubits (:math:`3 = \log_2 8`) and the target register of two qubits (:math:`m = 2`). For instance, for the
first four indices, the QROM operator acts as:

.. math::
     \begin{align}
     \text{QROM}|000\rangle|00\rangle &= |000\rangle|01\rangle \\
     \text{QROM}|001\rangle|00\rangle &= |001\rangle|11\rangle \\
     \text{QROM}|010\rangle|00\rangle &= |010\rangle|11\rangle \\
     \text{QROM}|011\rangle|00\rangle &= |011\rangle|00\rangle.
     \end{align}

We will now explain three different implementations of QROM: *Select*, *SelectSwap*, and an extension of *SelectSwap*.

Select
~~~~~~~

:class:`~.pennylane.Select` is an operator that prepares quantum states associated with indices. It is defined as:

.. math::

    \text{Select}|i\rangle|0\rangle = |i\rangle U_i|0\rangle =|i\rangle|\phi_i\rangle,

where :math:`|\phi_i\rangle` is the :math:`i`-th state we want to encode, generated by a known unitary :math:`U_i`.
QROM can be considered a special case of the *Select* operator where the encoded states are computational basis states.
Then the unitaries :math:`U_i` can be simply products of :math:`X` gates satisfying:

.. math::

    U_i|0\rangle =|b_i\rangle.

We use :class:`~.pennylane.BasisState` as a useful template for implementing the gates :math:`U_i`. Let's see how it could be written in code:

"""

import pennylane as qml
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0, 1, 2]
target_wires = [3, 4]

Ui = [qml.BasisState(int(bitstring, 2), target_wires) for bitstring in bitstrings]

dev = qml.device("default.qubit", shots=1)


# This line is included for drawing purposes only.
@partial(qml.devices.preprocess.decompose, stopping_condition=lambda obj: False, max_expansion=1)

@qml.qnode(dev)
def circuit(index):
    qml.BasisState(index, wires=control_wires)
    qml.Select(Ui, control=control_wires)
    return qml.sample(wires=target_wires)


qml.draw_mpl(circuit, style="pennylane")(3)
plt.show()

##############################################################################
# Now we can check that all the outputs are as expected:

for i in range(8):
    print(f"The bitstring stored in index {i} is: {circuit(i)}")


##############################################################################
# The outputs match the elements of our initial data list: :math:`[01, 11, 11, 00, 01, 11, 11, 00]`. Nice!
#
# The :class:`~.pennylane.QROM` template can be used to implement the previous circuit using directly the bitstring
# without having to calculate the :math:`U_i` gates:

import warnings
# This line will suppress ComplexWarnings for output visibility
warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0, 1, 2]
target_wires = [3, 4]


@partial(qml.compile, basis_set="CNOT")  # Line added for resource estimation purposes only.
@qml.qnode(dev)
def circuit(index):
    qml.BasisState(index, wires=control_wires)
    qml.QROM(bitstrings, control_wires, target_wires, work_wires=None)
    return qml.sample(wires=target_wires)


for i in range(8):
    print(f"The bitstring stored in index {i} is: {circuit(i)}")

##############################################################################
# Although this approach works correctly, the number of multicontrol gates is high — gates with a costly decomposition.
# Here we show the number of 1 and 2 qubit gates we use when decomposing the circuit:

print("Number of qubits: ", len(control_wires + target_wires))
print("One-qubit gates: ", qml.specs(circuit)(0)["resources"].gate_sizes[1])
print("Two-qubit gates: ", qml.specs(circuit)(0)["resources"].gate_sizes[2])

##############################################################################
# You can learn more about these resource estimation methods in
# the `PennyLane documentation <https://docs.pennylane.ai/en/stable/code/qml_resource.html>`__.
# There are numerous works that attempt to simplify this, of which
# we highlight reference [#unary]_, which introduces an efficient technique using measurements in the middle
# of the circuit. Another clever approach was introduced in [#selectSwap]_ , with a smart structure known as *SelectSwap*,
# which we describe below.
#
# SelectSwap
# ~~~~~~~~~~
# The goal of the *SelectSwap* construction is to trade depth of the circuit for width. That is, using multiple auxiliary qubits,
# we reduce the circuit depth required to build the QROM. Before we get into how it works, let's show you how easy it is to use:
# we simply add ``work_wires`` to the code we had previously.
#

bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0, 1, 2]
target_wires = [3, 4]
work_wires = [5, 6]


@partial(qml.compile, basis_set="CNOT") 
@qml.qnode(dev)
def circuit(index):
    qml.BasisState(index, wires=control_wires)
    #  added work wires below
    qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean=False)
    return qml.sample(wires=control_wires + target_wires + work_wires)

print("Number of qubits: ", len(control_wires + target_wires + work_wires))
print("One-qubit gates: ", qml.specs(circuit)(0)["resources"].gate_sizes[1])
print("Two-qubit gates: ", qml.specs(circuit)(0)["resources"].gate_sizes[2])

##############################################################################
# The number of 1 and 2 qubit gates is significantly reduced!
#
# Internally, the main idea of this approach is to organize the :math:`U_i` operators into two dimensions,
# whose positions will be determined by a column index :math:`c` and a row index :math:`r`.
#
# .. figure:: ../_static/demonstration_assets/qrom/select_swap.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Following this structure, for instance, the :math:`U_5` operator (or :math:`101` in binary) is in column :math:`2` and row :math:`1` (zero-based indexing):
#
# .. figure:: ../_static/demonstration_assets/qrom/indixes_qrom.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# In order to load the desired bitstring in the target wires, we use two building blocks in the construction:
#
# - **Select block**: Loads the :math:`c`-th column in the target and work wires.
# - **Swap block**: Swaps the :math:`r`-th row to the target wires.
#
#
# Let's look at an example by assuming we want to load in the target wires the bitstring with
# the index :math:`5`, i.e., :math:`U_5|0\rangle = |b_5\rangle`.
#
# .. figure:: ../_static/demonstration_assets/qrom/example_selectswap.jpeg
#    :align: center
#    :width: 85%
#    :target: javascript:void(0)
#
# Now we run the circuit with our initial data list: :math:`[01, 11, 11, 00, 01, 11, 11, 00]`.

index = 5
output = circuit(index)
print(f"control wires: {output[:3]}")
print(f"target wires: {output[3:5]}")
print(f"work wires: {output[5:7]}")


##############################################################################
# As expected, :math:`|b_5\rangle = |11\rangle` is loaded in the target wires.
# Note that with more auxiliary qubits we could make larger groupings of bitstrings reducing the depth of the
# *Select* operator. Below we show an example with two columns and four rows:
#
# .. figure:: ../_static/demonstration_assets/qrom/select_swap_4.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# The QROM template will put as many rows as possible using the ``work_wires`` we pass.
# Let's check how it looks in PennyLane:

bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0, 1, 2]
target_wires = [3, 4]
work_wires = [5, 6, 7, 8, 9, 10, 11, 12]


# Line added for drawing purposes only
@partial(qml.devices.preprocess.decompose, stopping_condition=lambda obj: False, max_expansion=2)
@qml.qnode(qml.device("default.qubit", shots=1))
def circuit(index):
    qml.BasisState(index, wires=control_wires)
    qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean=False)
    return qml.sample(wires=target_wires), qml.sample(wires=target_wires)


qml.draw_mpl(circuit, style="pennylane")(0)
plt.show()


##############################################################################
# The circuit matches the one described above.
#
# Reusable qubits
# ~~~~~~~~~~~~~~~~
#
# The above approach has a drawback. The work wires have been altered, i.e., after applying the operator they have not
# been returned to state :math:`|00\rangle`. This could cause unwanted behaviors, but in PennyLane it can be easily solved
# by setting the parameter ``clean = True``.


bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0, 1, 2]
target_wires = [3, 4]
work_wires = [5, 6]


@qml.qnode(dev)
def circuit(index):
    qml.BasisState(index, wires=control_wires)
    qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean=True)
    return qml.sample(wires=target_wires + work_wires)


for i in range(8):
    print(f"The bitstring stored in index {i} is: {circuit(i)[:2]}")
    print(f"The work wires for that index are in the state: {circuit(i)[2:4]}\n")


##############################################################################
# All the work wires have been reset to the zero state.
#
# To achieve this, we follow the technique shown in [#cleanQROM]_, where the proposed circuit (with :math:`R` rows) is as follows:
#
# .. figure:: ../_static/demonstration_assets/qrom/clean_version_2.jpeg
#    :align: center
#    :width: 90%
#    :target: javascript:void(0)
#
# To see how this circuit works, let's suppose we want to load the bitstring :math:`b_{cr}` in the target wires, where :math:`b_{cr}`
# is the bitstring whose operator :math:`U` is placed in the :math:`c`-th column and :math:`r`-th row in the two-dimensional representation shown in the *Select* block.
# We can summarize the idea in a few simple steps.
#
# 0. **Initialize the state.** We create the state:
#
# .. math::
#       |c\rangle |r\rangle |0\rangle |0\rangle \dots |0\rangle.
#
# 1. **A uniform superposition is created in the r-th register of the work wires**. To do this, we put the Hadamards in the target wires and move it to the :math:`r`-th row with the *Swap* block:
#
# .. math::
#       |c\rangle |r\rangle |0\rangle |0\rangle \dots |+\rangle_r \dots |0\rangle.
#
# 2. **The Select block is applied.** This loads the whole :math:`c`-th column in the registers. Note that in the :math:`r`-th position, the *Select* has no effect since the state :math:`|+\rangle` is not modified by :math:`X` gates:
#
# .. math::
#       |c\rangle |r\rangle |b_{c0}\rangle |b_{c1}\rangle \dots |+\rangle_r \dots |b_{c(R-1)}\rangle.
#
#
# 3. **The Hadamard gate is applied to the r-th register of the work wires.** This returns that register to the zero state. The two *Swap* blocks and the Hadamard gate applied to the target wires achieve this:
#
# .. math::
#       |c\rangle |r\rangle |b_{c0}\rangle |b_{c1}\rangle \dots |0\rangle_r \dots |b_{c(R-1)}\rangle.
#
# 4. **Select block is applied.** Thanks to this, we clean the used registers. That is because loading the bitstring twice in the same register leaves the state as :math:`|0\rangle` since :math:`X^2 = \mathbb{I}`. On the other hand, the bitstring :math:`|b_{cr}\rangle` is loaded in the :math:`r` register:
#
# .. math::
#       |c\rangle |r\rangle |0\rangle |0\rangle \dots |b_{cr}\rangle_r \dots |0\rangle.
#
# 5. **Swap block is applied.** With this, we move :math:`|b_{cr}\rangle` that is encoded in the r-th row to the target wires:
#
# .. math::
#       |c\rangle |r\rangle |b_{cr}\rangle |0\rangle \dots |0\rangle_r \dots |0\rangle.
#
# The desired bitstring has been encoded in the target wires and the rest of the qubits have been left in the zero state.
#
# Conclusion
# ----------
#
# By implementing various versions of the QROM operator, such as *Select* and *SelectSwap*, we can optimize quantum circuits
# for enhanced performance and scalability. These methods improve the efficiency of
# state preparation [#StatePrep]_ techniques. After all, state preparation is a special case of data encoding, where the data are the coefficients that define the state.
# QROM methods are particularly attractive for large-scale quantum computing due to their superior asymptotic efficiency.
# This makes them an indispensable tool for developing new algorithms.
#
# References
# ----------
#
# .. [#unary]
#
#       Ryan Babbush, Craig Gidney, Dominic W. Berry, Nathan Wiebe, Jarrod McClean, Alexandru Paler, Austin Fowler, and Hartmut Neven,
#       "Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity,"
#       `Physical Review X, 8(4), 041015 (2018). <http://dx.doi.org/10.1103/PhysRevX.8.041015>`__, 2018
#
# .. [#selectSwap]
#
#       Guang Hao Low, Vadym Kliuchnikov, and Luke Schaeffer,
#       "Trading T-gates for dirty qubits in state preparation and unitary synthesis",
#       `arXiv:1812.00954 <https://arxiv.org/abs/1812.00954>`__, 2018
#
# .. [#cleanQROM]
#
#       Dominic W. Berry, Craig Gidney, Mario Motta, Jarrod R. McClean, and Ryan Babbush,
#       "Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank Factorization",
#       `Quantum 3, 208 <http://dx.doi.org/10.22331/q-2019-12-02-208>`__, 2019
#
# .. [#StatePrep]
#
#       Lov Grover and Terry Rudolph,
#       "Creating superpositions that correspond to efficiently integrable probability distributions",
#       `arXiv:quant-ph/0208112 <https://arxiv.org/abs/quant-ph/0208112>`__, 2002
#
# About the authors
# -----------------
