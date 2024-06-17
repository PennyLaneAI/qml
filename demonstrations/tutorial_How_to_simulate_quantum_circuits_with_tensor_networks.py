r"""How to simulate quantum circuits with tensor networks
====================================================================

Tensor networks are a powerful computational tool for simulating quantum circuits.
They provide a way to represent quantum states and operations in a compact form. 
Unlike the state vector approach, tensor networks are particularly useful for large-scale simulations of quantum circuits.

Here, we demonstrate how to simulate quantum circuits using the ``default.tensor`` device in PennyLane.
This simulator is based on `quimb <https://quimb.readthedocs.io/en/latest/>`__, a Python library for tensor network manipulations. 
The ``default.tensor`` device is convenient for simulations with tens, hundreds, or even thousands of qubits.
Other devices based on the state vector approach may be more suitable for small circuits 
since the overhead of tensor network contractions can be significant.

TODO: Insert figure

"""

######################################################################
# ...
#
# About the authors
# -----------------
