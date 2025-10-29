r"""

Lattice Surgery
===============

Lattice surgery is a way to fault-tolerantly perform operations on two-dimensional topological QEC codes
such as the surface code. These concepts can be generalized to other codes such as color codes or folded surface codes.
This demo is serving the purpose of being a refresher or first intro to the basics of surface code quantum computing
with lattice surgery.

.. figure:: ../_static/demonstration_assets/block_encoding/thumbnail_Block_Encodings_Matrix_Oracle.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

abstract

Before heading into the topic of lattice surgery [#latticesurgery], we first need to understand the surface code [#surfacecode]_.

Error Detection with stabilizers and syndromes
----------------------------------------------

Let us start simple and construct a very basic system of two qubits ``wires=["a", "b"]`` 
with measurement operators :math:`\hat{X}_a \hat{X}_b` and :math:`\hat{Z}_a \hat{Z}_b`.
These two operators have the joint eigenbasis consisting of the four Bell states with the following eigenvalues:

.. list-table::
   :widths: 40 40 40
   :header-rows: 1

   * - :math:`\hat{Z}_a \hat{Z}_b`
     - :math:`\hat{X}_a \hat{X}_b`
     - state
   * - +1
     - +1
     - :math:`|00\rangle + |11\rangle`
   * - +1
     - -1
     - :math:`|00\rangle - |11\rangle
   * - -1
     - +1
     - :math:`|01\rangle + |10\rangle
   * - -1
     - -1
     - :math:`|01\rangle - |10\rangle

If we start in one of the eigenstates, e.g. :math:`|00\rangle + |11\rangle`, we can repeatedly measure
:math:`\hat{X}_a \hat{X}_b` and :math:`\hat{Z}_a \hat{Z}_b`
without perturbing the system. The measurement result will always be the same, :math:`(+1, +1)`, 
in the case of :math:`|00\rangle + |11\rangle`. We call this measurement result a *syndrome*.

The unperturbing measurements are possible because :math:`\hat{X}_a \hat{X}_b` and :math:`\hat{Z}_a \hat{Z}_b`
commute. We call such a group of commuting observables as *stabilizers*. 
In particular, they allow us to detect errors via their syndrome:
Whenever there is a single qubit error is occuring, be it a bit flip (:math:`\hat{X}`) or a phase shift (:math:`\hat{Z}`)
it maps one of the eigenstates to another eigenstate, changing the syndrome.

The overall idea is then the following: we encode a logical qubit state in two of the eigenstates of the two qubits, e.g.
:math:`|0\rangle_L = |00\rangle_{ab} + |11\rangle_{ab}` and :math:`|1\rangle_L = |00\rangle_{ab} - |11\rangle_{ab}`.
We then repeatedly measure the syndrome, which, in an ideal scenario, remains unchanged when the qubits are idling.
If we detect a change in the syndrome, we know an error has occured.

While this very simple construction allows us to *detect* errors, it is lacking the capability to *correct* errors.
In order to do so, we need something more elaborate like the surface code.

Error Correction using the Surface Code
---------------------------------------

The most simple planar surface code consists of a rectangular lattice of data qubits
(open circles :math:`\circ`) and syndrome qubits (full circles :math:`\bullet`):

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

These *physical* qubits need to be capable of universal quantum computing (CNOTs and single qubit rotations),
where errors are inevitably occuring - though, ideally, at low error rates.
Data qubits are storing the quantum state of the overall system, which comprises a logical qubit state.
The sole purpose of syndrome qubits is to *indirectly* measure the stabilizers of the planar surface code on
the data qubits. Note that in many higher level depictions of surface codes, syndrome qubits are not shown and the view is rotated by :math:`45^\circ` (see image below).

The stabilizers or the planar surface code are 


.. list-table::
   :widths: 40 60 60
   :header-rows: 1

   * - Eigenvalue
     - :math:`\hat{Z}_a \hat{Z}_b \hat{Z}_c \hat{Z}_d`
     - :math:`\hat{X}_a \hat{X}_b \hat{X}_c \hat{X}_d`
   * - +1
     - :math:`|0000\rangle`
     - :math:`|++++\rangle`
   * - +1
     - :math:`|0011\rangle`
     - :math:`|++--\rangle`
   * - +1
     - :math:`|0110\rangle`
     - :math:`|+--+\rangle`
   * - +1
     - :math:`|1100\rangle`
     - :math:`|--++\rangle`
   * - +1
     - :math:`|1001\rangle`
     - :math:`|-++-\rangle`
   * - +1
     - :math:`|0101\rangle`
     - :math:`|+-+-\rangle`
   * - +1
     - :math:`|1010\rangle`
     - :math:`|-+-+\rangle`
   * - +1
     - :math:`|1111\rangle`
     - :math:`|----\rangle`
   * - -1
     - :math:`|0001\rangle`
     - :math:`|+++-\rangle`
   * - -1
     - :math:`|0010\rangle`
     - :math:`|++-+\rangle`
   * - -1
     - :math:`|0100\rangle`
     - :math:`|+-++\rangle`
   * - -1
     - :math:`|1000\rangle`
     - :math:`|-+++\rangle`
   * - -1
     - :math:`|1110\rangle`
     - :math:`|---+\rangle`
   * - -1
     - :math:`|1101\rangle`
     - :math:`|--+-\rangle`
   * - -1
     - :math:`|1011\rangle`
     - :math:`|-+--\rangle`
   * - -1
     - :math:`|0111\rangle`
     - :math:`|+---\rangle`

"""

import pennylane as qml


##############################################################################
#
# Universal Gate set in the surface code using lattice surgery
# ------------------------------------------------------------
# 
# asd

##############################################################################
#

##############################################################################
#

##############################################################################
#

##############################################################################
#


##############################################################################
# text
#
# Conclusion
# ----------
# text
#
# References
# ----------
#
# .. [#surfacecode]
#
#     Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland,
#     "Surface codes: Towards practical large-scale quantum computation",
#     `arXiv:1208.0928 <https://arxiv.org/abs/1208.0928>`__, 2012
#
#
# .. [#latticesurgery]
#
#     Dominic Horsman, Austin G. Fowler, Simon Devitt, Rodney Van Meter,
#     "Surface code quantum computing by lattice surgery",
#     `arXiv:1111.40226 <https://arxiv.org/abs/1111.4022>`__, 2011
#
# .. [#Chamberland]
#
#     Christopher Chamberland, Earl T. Campbell
#     "Universal quantum computing with twist-free and temporally encoded lattice surgery",
#     `arXiv:2109.02746 <https://arxiv.org/abs/2109.02746>`__, 2021
#
# .. [#Chamberland]
#
#     Christopher Chamberland, Earl T. Campbell
#     "Universal quantum computing with twist-free and temporally encoded lattice surgery",
#     `arXiv:2109.02746 <https://arxiv.org/abs/2109.02746>`__, 2021
#
# .. [#Litinski]
#
#     Daniel Litinski
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
#     `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.
#
# .. [#Fowler]
#
#     Austin G. Fowler, Craig Gidney
#     "Low overhead quantum computation using lattice surgery"
#     `arXiv:1808.06709 <https://arxiv.org/abs/1808.06709>`__, 2018.
#
#
# Disclaimer
# ----------
# This demo is a Frankenstein of the two seminal papers on surface code quantum computing [#surfacecode]_ and [#latticesurgery]_.
# First two sections follow closely the intro sections of reference [#surfacecode]_, and the final section is inspired by [#latticesurgery]_.

