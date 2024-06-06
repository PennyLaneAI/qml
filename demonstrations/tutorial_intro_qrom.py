r"""Intro to QROM
=============================================================

Managing data is an indispensable task on any computer. Quantum computers are no different and getting this
done efficiently plays a crucial role in fields such as QML or can even be useful in search algorithms.
In this demo we will introduce the concept of QROM, the data structure that allows us to work towards this task.

QROM
-----

Quantum Read-Only Memory (QROM) is an operator that allows us to load classical data into a quantum computer
associated with indeces. This data is represented as a bitstring (list of 0s and 1s) and the operator can be defined as:

.. math::

    \text{QROM}|i\rangle|0\rangle = |i\rangle|b_i\rangle,

where :math:`|b_i\rangle` is the bitstring associated with the index :math:`i`.
Suppose our data consists of eight bit-strings :math:`[11, 01, 11, 00, 10, 10, 11, 00]`. Then, the index register will consist of three
qubits (:math:`\log_2 8`) and the target register of two qubits (length of the bit-strings). Following the same example,
:math:`\text{QROM}|010\rangle|00\rangle = |010\rangle|11\rangle`, since the bit-string associated with index :math:`2` is :math:`11`.
We will show three different implementations of this operator: Select, SelectSwap and an advanced version of the
last one.

Select
~~~~~~~

Select is an operator that prepares quantum states associated with indices. It is defined as:

.. math::

    \text{Sel}|i\rangle|0\rangle = |i\rangle|\phi_i\rangle,

where :math:`|\phi_i\rangle` is the i-th state we want to encode generated by a known-gate :math:`U_i`.
Since the bitstrings can be seen as a particular quantum state, we particularize this operator to the QROM case.
For the following example we are going to use :class:`~.pennylane.BasisEmbedding` as :math:`U_i`, and
the :class:`~.pennylane.Select` template provided by PennyLane.

"""

import pennylane as qml
from functools import partial

control_wires = [0,1,2,3]
target_wires = [4]

bitstrings = ["0", "1", "1", "0", "1", "1", "1", "0", "0", "1", "1", "0", "1", "1", "1", "0"]
Ui = [qml.BasisEmbedding(int(bitstring, 2), target_wires) for bitstring in bitstrings]

dev = qml.device("default.qubit", shots = 1)

# I put this line so that the circuit can be visualized more clearly afterwards.
@partial(qml.devices.preprocess.decompose,
         stopping_condition = lambda obj: False,
         max_expansion=1)
@qml.qnode(dev)
def circuit(index):
    qml.BasisEmbedding(index, wires=control_wires)
    qml.Select(Ui, control=control_wires)
    return qml.sample(wires=target_wires)

##############################################################################
# Once we have defined the circuit, we can draw it and check that the outputs are as expected.

import matplotlib.pyplot as plt

qml.draw_mpl(circuit, style = "pennylane")(0)
plt.show()

for i in range(16):
    print(f"The bitstring stored in the {i}-index is: {circuit(i)}")


##############################################################################
# Nice, you can see that the outputs match the elements of our initial data list.
#
# Although the algorithm works correctly, we can see that the number of multicontrol gates is high.
# The decomposition of these gates is expensive and there are numerous works that attempt to simplify this.
# We can highlight the work [google] which introduces an efficient technique making use of measurements in the middle
# of the circuit. Another clever approach was introduced in [here], with a smart structure known as SelectSwap.
#
# SelectSwap
# ~~~~~~~~~~
# The SelectSwap goal is to trade depth for width of the circuit. That is, using auxiliary qubits,
# reduce the number of gates required to build the QROM. We can detail the algorithm in two steps.
#
# .. figure:: ../_static/demonstration_assets/qrom/select_swap.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# First we use the auxiliary qubits to store more than one bitstring per column.
# In this way, we reduce the number of operators that Select requires.
# The control qubits of the Select block determine in which column is the bitstring we want to load.
# Secondly, the swap block detects the row where searched bitstring is located and
# swap it to the target wires.
#
# Note that with more auxiliary qubits we could make larger groupings of bitstrings reducing more the workload off the
# Select operator.
#
# .. figure:: ../_static/demonstration_assets/qrom/select_swap_4.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Reusable SelectSwap
# ~~~~~~~~~~~~~~~~~~~
# The above approach has a drawback. The work wires have been altered, i.e., after applying the operator they have not
# been returned to state :math:`|0\rangle`. This can cause unwanted behaviors, so we will present the technique shown
# in [paper] to solve this.
#
# .. figure:: ../_static/demonstration_assets/qrom/clean_version.jpeg
#    :align: center
#    :width: 90%
#    :target: javascript:void(0)
#
# Following the same idea as before, the control wires of the select block will choose the column :math:`c` where the
# bitstring is located and those of the swap block the row :math:`r`.
# We can summarize the idea of why the circuit works in a few simple steps:
#
# 1. **We start by generating the uniform superposition on the r-th register**.
#
# .. math::
#       |c\rangle |r\rangle |0\rangle \dots |+\rangle_r \dots |0\rangle
#
# 2. **We apply the select block.** We denote by :math:`b_{cr}` the bitstring of column :math:`c` and row :math:`r`. Note that in the :math:`r`-th position, the Select has no effect since this state is not modified by :math:`X` gates.
#
# .. math::
#       |c\rangle |r\rangle |b_{c0}\rangle \dots |+\rangle \dots |b_{cR}\rangle
#
#
# 3. **We apply the Hadamard's in r-th register.**
#
# .. math::
#       |c\rangle |r\rangle |b_{c0}\rangle \dots |0\rangle \dots |b_{cR}\rangle
#
# 4. **We apply select again to the state.** Note that applying Select twice on a state is equivalent to the identity.
#
# .. math::
#       |c\rangle |r\rangle |0\rangle \dots |b_{cr}\rangle \dots |0\rangle
#
# That's it! With a last swap we have managed to load the bitstring of column :math:`c` and row :math:`r` in the target wires.
#
# QROM in PennyLane
# -----------------
# Now it is time to show the potential of Pennylane and demonstrate the two methods mentioned above.
# To do this, we encode the same bitstrings, using the first SelectSwap approach. We are going to use three work wires
# to store four blocks per column:


control_wires = [0,1,2,3]
target_wires = [4]
work_wires = [5,6,7]

# This function is included for drawing purposes only.
def my_stop(obj):
  if obj.name in ["CSWAP", "BasisEmbedding", "Hadamard"]:
    return True
  return False

@partial(qml.devices.preprocess.decompose, stopping_condition = my_stop, max_expansion=2)
@qml.qnode(qml.device("default.qubit", shots = 1))
def circuit(ind):

  qml.BasisEmbedding(ind, wires = control_wires)

  qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean = False)

  return qml.sample(wires = target_wires)

qml.draw_mpl(circuit, style = "pennylane")(0)
plt.show()

for ind in range(16):
  print(f"The index {ind} is storing the state {circuit(ind)}")

##############################################################################
# The outputs are the same as before, but the circuit is much simpler.
# If we want to clean the work wires, we could set the ``clean`` attribute of QROM to ``True``.
# Let's see how the circuit looks like:

@partial(qml.devices.preprocess.decompose, stopping_condition = my_stop, max_expansion=2)
@qml.qnode(qml.device("default.qubit", shots = 1))
def circuit(ind):

  qml.BasisEmbedding(ind, wires = control_wires)

  qml.QROM(bitstrings, control_wires, target_wires, work_wires, clean = True)

  return qml.sample(wires = target_wires)

qml.draw_mpl(circuit, style = "pennylane")(1)
plt.show()

##############################################################################
# Beautiful! The circuit is more complex, but the work wires are clean.
# As a curiosity, this template works with work wires that are not initialized to zero.
# You could use other qubits in your circuit without altering their state.
#
#
# Conclusion
# ----------
#
#
# About the author
# ----------------
