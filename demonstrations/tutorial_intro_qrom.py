r"""Intro to QROM
=============================================================

Managing data is a crucial task on any computer, and quantum computers are no exception. Efficient data management is vital in quantum machine learning, search algorithms, and state preparation.
In this demonstration, we will introduce the concept of a Quantum Read-Only Memory (QROM), a data structure designed to load classical data on a quantum computer.
You will also see how easy it is to use this operator in PennyLane through the :class:`~.pennylane.QROM` template.

QROM
-----

The QROM is an operator that allows us to load classical data into a quantum computer. Data is represented as a collection of bitstrings (list of 0s and 1s) that we denote by :math:`b_1, b_2, \ldots, b_N`. The QROM operator is then defined as:

.. math::

    \text{QROM}|i\rangle|0\rangle = |i\rangle|b_i\rangle,

where :math:`|b_i\rangle` is the bitstring associated with the index :math:`i`.

For example, suppose our data consists of four bit-strings, each with three bits: :math:`[011, 101, 111, 100]`. Then, the index register will consist of two
qubits (:math:`2 = \log_2 4`) and the target register of three qubits (length of the bit-strings). In this case, the QROM operator acts as:

.. math::
     \begin{align}
     \text{QROM}|00\rangle|000\rangle &= |00\rangle|011\rangle \\
     \text{QROM}|01\rangle|000\rangle &= |01\rangle|101\rangle \\
     \text{QROM}|10\rangle|000\rangle &= |10\rangle|111\rangle \\
     \text{QROM}|11\rangle|000\rangle &= |11\rangle|100\rangle
     \end{align}

We will now explain three different implementations of QROM: Select, SelectSwap, and an extension of SelectSwap.

Select
~~~~~~~

:class:`~.pennylane.Select` is an operator that prepares quantum states associated with indices. It is defined as:

.. math::

    \text{Select}|i\rangle|0\rangle = |i\rangle U_i|0\rangle =|i\rangle|\phi_i\rangle,

where :math:`|\phi_i\rangle` is the :math:`i`-th state we want to encode, generated by a known unitary :math:`U_i`.
QROM can be considered a special case of the Select operator where the encoded states are computational basis states.
Then the unitaries :math:`U_i` can be simply :math:`X` gates that determine whether each bit is a :math:`0` or a :math:`1`.
We use :class:`~.pennylane.BasisEmbedding` as a useful template for preparing bitstrings, it places the :math:`X` gates
in the right position. Let's use a longer string for the following example:

"""

import pennylane as qml
from functools import partial
import matplotlib.pyplot as plt


bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0,1,2]
target_wires = [3,4]

Ui = [qml.BasisEmbedding(int(bitstring, 2), target_wires) for bitstring in bitstrings]

dev = qml.device("default.qubit", shots = 1)

# This line is included for drawing purposes only.
@partial(qml.devices.preprocess.decompose,
         stopping_condition = lambda obj: False,
         max_expansion=1)

@qml.qnode(dev)
def circuit(index):
    qml.BasisEmbedding(index, wires=control_wires)
    qml.Select(Ui, control=control_wires)
    return qml.sample(wires=target_wires)

qml.draw_mpl(circuit, style = "pennylane")(0)
plt.show()
##############################################################################
# Now we can check that all the outputs are as expected:

for i in range(8):
    print(f"The bitstring stored in index {i} is: {circuit(i)}")


##############################################################################
# Nice, the outputs match the elements of our initial data list: :math:`[01, 11, 11, 00, 01, 11, 11, 00]`.
#
# Although the algorithm works correctly, the number of multicontrol gates is high.
# The decomposition of these gates is expensive and there are numerous works that attempt to simplify this.
# We highlight reference [#unary]_ which introduces an efficient technique using measurements in the middle
# of the circuit. Another clever approach was introduced in [#selectSwap]_ , with a smart structure known as SelectSwap, which we describe below.
#
# SelectSwap
# ~~~~~~~~~~
# The goal of the SelectSwap construction is to trade depth for width. That is, using multiple auxiliary qubits,
# we reduce the circuit depth required to build the QROM. The main idea is to organize the data in two dimensions,
# with each bitstring labelled by a column index :math:`c` and a row index :math:`r`.
# The control qubits of the Select block determine the column :math:`c`, while the
# control qubits of the swap block are used to specify the row index :math:`r`.
#
# .. figure:: ../_static/demonstration_assets/qrom/select_swap.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Let's look at an example by assuming we want to load in the target wires the bitstring with
# the index :math:`5`.
# For it, we put as input in the control wires the state :math:`|101\rangle` (5 in binary), where the first two bits refer to the
# index :math:`c = |10\rangle` and the last one to the index :math:`r = |1\rangle`.  After applying the Select block, we
# obtain :math:`|101\rangle|01\rangle|11\rangle`, loading the bitstrings :math:`b_4` and :math:`b_5` respectively.
# Since the third
# control qubit (i.e., :math:`r`) is a :math:`|1\rangle`, it will activate the swap block, generating the state :math:`|101\rangle|11\rangle|01\rangle`
# loading the bitstring :math:`b_5` in the target register.
#
# Note that with more auxiliary qubits we could make larger groupings of bitstrings reducing the workload of the
# Select operator. Below we show an example with two columns and four rows:
#
# .. figure:: ../_static/demonstration_assets/qrom/select_swap_4.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Using the same example, we have that :math:`c = |1\rangle` and :math:`r = |01\rangle`. In this case, the columns are
# determined by a single index but we need two indexes for the rows. We invite you to check that :math:`b_5` is actually
# loaded in the target wires.
#
#
# Reusable qubits
# ~~~~~~~~~~~~~~~~~
#
# The above approach has a drawback. The work wires have been altered, i.e., after applying the operator they have not
# been returned to state :math:`|0\rangle`. This can cause unwanted behaviors, so we will present the technique shown
# in [#cleanQROM]_ to solve this.
#
# .. figure:: ../_static/demonstration_assets/qrom/clean_version_2.jpeg
#    :align: center
#    :width: 90%
#    :target: javascript:void(0)
#
# To see how this circuit works, let's suppose we want to load the bitstring :math:`b_{cr}`, in the target wires.
# We can summarize the idea in a few simple steps:
#
# 1. **We start by generating the uniform superposition on the r-th register**. To do this, we put the Hadamard in the target wires and moved it to the :math:`r` -row with the swap block.
#
# .. math::
#       |c\rangle |r\rangle |0\rangle \dots |+\rangle_r \dots |0\rangle
#
# 2. **We apply the Select block.** Note that in the :math:`r`-th position, the Select has no effect since the state :math:`|+\rangle` is not modified by :math:`X` gates.
#
# .. math::
#       |c\rangle |r\rangle |b_{c0}\rangle \dots |+\rangle_r \dots |b_{cR}\rangle
#
#
# 3. **We apply the Hadamard's in r-th register.** The two swap blocks and the Hadamard gate in target wires achieve this.
#
# .. math::
#       |c\rangle |r\rangle |b_{c0}\rangle \dots |0\rangle_r \dots |b_{cR}\rangle
#
# 4. **We apply Select again to the state.** Note that loading the bitstring twice in the same register leaves the state as :math:`|0\rangle`. (:math:`X^2 = \mathbb{I}`)
#
# .. math::
#       |c\rangle |r\rangle |0\rangle \dots |b_{cr}\rangle_r \dots |0\rangle
#
# That's it! With a last swap we have managed to load the bitstring of column :math:`c` and row :math:`r` in the target wires.
#
# QROM in PennyLane
# -----------------
# Coding a QROM circuit from scratch can be painful, but with the help of PennyLane you can do it in just one line of code.
# Let's see an example where we encode longer bitstrings and we will use enough work wires to group four bitstrings per column:

bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

control_wires = [0, 1, 2]
target_wires = [3, 4]
work_wires = [5, 6, 7, 8, 9, 10]

@qml.qnode(qml.device("default.qubit", shots = 1))
def circuit(index):

    qml.BasisEmbedding(index, wires=control_wires)

    qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean = False)

    return qml.sample(wires = target_wires), qml.sample(wires = work_wires)

for i in range(8):
    print(f"The bitstring stored in index {i} is: {circuit(i)[0]}")
    print(f"The work wires for that index are in the state: {circuit(i)[1]}\n")

##############################################################################
# .The list has been correctly encoded. However, we can see that the auxiliary qubits have been altered.
#
# If we want to use the approach that cleans the work wires, we could set the ``clean`` attribute of QROM to ``True``.
# Let's see how the circuit looks like:

bitstrings = ["01", "11", "11", "00", "01", "11", "11", "00"]

@qml.qnode(qml.device("default.qubit", shots = 1))
def circuit(index):

    qml.BasisEmbedding(index, wires=control_wires)

    qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean = True)

    return qml.sample(wires=target_wires), qml.sample(wires=work_wires)


for i in range(8):
    print(f"The bitstring stored in index {i} is: {circuit(i)[0]}")
    print(f"The work wires for that index are in the state: {circuit(i)[1]}\n")

##############################################################################
# Great! As you can see the work wires have been cleaned and all versions worked correctly.
# As a curiosity, this template works with work wires that are not initialized to zero.
#
#
#
# Conclusion
# ----------
#
# By implementing various versions of the QROM operator, such as Select and SelectSwap, we optimize quantum circuits
# for enhanced performance and scalability. These methods improve the efficiency of
# state preparation [#StatePrep]_ techniques by reducing the number of required gates, which we recommend you explore.
# As the availability of qubits increases, the relevance of these methods will grow making this operator an
# indispensable tool for developing new algorithms and an interesting field for further study.
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
# About the author
# ----------------
