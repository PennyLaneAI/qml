r"""Intro to QROM
=============================================================

Storing and loading data is an indispensable task on any computer. Quantum computers are no different and getting this
done efficiently plays a crucial role in fields such as QML or can even be useful in search algorithms.
In this demo we will introduce the concept of QROM, the data structure that allows us to work towards this task.

QROM
-----

Quantum Read-Only Memory (QROM) is an operator that allows us to load classical data into a quantum computer
associated with indeces. This data is represented as a bitstring (list of 0s and 1s) and the operator can be defined as:

.. math::

    \text{QROM}|i\rangle|0\rangle = |i\rangle|b_i\rangle,

where :math:`|b_i\rangle` is the bitstring associated with the index :math:`i`.
Suppose our data consists of four bit-strings: :math:`[110, 010, 111, 000]`. Then, the index register will consist of two
qubits (:math:\`log_2 4`) and the target register of three qubits (length of the bit-strings). Following the same example:

.. math::

    \text{QROM}|10\rangle|000\rangle = |10\rangle|111\rangle,

since the bit-string associated with index :math:`2` is :math:`111`.
We will show three different implementations of this operator: Select, SelectSwap and an advanced version of the
last one.

Select
~~~~~~~

- Introduce the Select operator and how it can be particularized to define a QROM.
- Draw circuit (code example).
- Talk about complexity of multicontrols. Here we can mention the google paper as a strategy to reduce this complexity.

Select SWAP
~~~~~~~~~~~~

- new strategy, storing bitstrings in parallel on different qubits.
- Present idea of Select Swap. (no code)

.. figure:: ../_static/demonstration_assets/qrom/qrom.gif
    :align: center
    :width: 50%
    :target: javascript:void(0)

- Problem: auxiliary qubits are not reusable.

reusable Select SWAP
~~~~~~~~~~~~~~~~~~~~~

- (There is no official name for this version of select swap. This is a name I have chosen)
- Present the template of the circuit. (picture, no code)
- Mathematical intuition (somewhat elegant)
- final curious fact: aux qubits no need to be initialize to zero. It will work with any arbitrary input.

QROM in Pennylane
-----------------

- Show an example to visually see Select SWAP with the two variants


Conclusion
-----------------

"""

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/juan_miguel_arrazola.txt
