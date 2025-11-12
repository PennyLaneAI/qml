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

Topological quantum error correction codes like the surface code [#surfacecode]_ protect quantum information of logical qubits by
ensuring the underlying physical qubits are kept in the code space, defined by the :math:`+1` eigenspace of a set of stabilizer measurements.
A logical qubit is then represented by a patch of physical qubits, called data qubits, on a square lattice like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit1.png
    :align: center
    :width: 30%
    :target: javascript:void(0)
  
Each intersection of a line corresponds to a data qubit; so :math:`5 \times 5` in total here. On yellow surfaces, we continually measure X stabilizers, which is simply the :math:`X_a X_b X_c X_d` operator
on the four data qubits in the corners. Similarly, Z stabilizers are measured on white surfaces. These measurements are performed via another kind of qubit, 
sometimes called measurement qubit or syndrome qubit, that is placed in the center of each square surface (not shown here, but see below).

Note that this representation of a qubit is in the so-called rotated surface code due to its :math:`45^\circ` rotation with respect to the original planar surface code.
Here we indicate the underlying data and syndrome qubits with black and red dots, respectively. 
The vertical lines do not correspond to physical connections, but are a mere guide to the eye, and also explain why the syndromes are sometimes
referred to as vertex and face syndromes.

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

There are different ways to perform logical operations like a CNOT gate on such a code. The most straight-forward way to compute a logical CNOT is by performing physical CNOT gates between the data qubits
of each logical qubit patch, indicated here by the red crosses:

.. figure:: ../_static/demonstration_assets/lattice_surgery/transversal_cnot.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is called a transversal operation, which is problematic as it requires non-local physical connections between data qubits.
Most quantum hardware architectures do not allow for this.
Instead, CNOT gates can be performed non-transversally via braiding [#braiding]_, a concept commonly encountered in algebraic topology.
In this setting, qubits are encoded by defects in the code, and operations via continuous deformations of the code.
However, defect based qubits suffer from requiring significantly more physical qubits per logical qubit.

This is where lattice surgery comes into play [#latticesurgery]_, as it has been shown to enable error-corrected logical operations
with significantly lower space resources with comparable time requirements [#Fowler]_ [#Litinski]_ [#Chamberland]_. The fundamental operations in lattice
surgery are discontinuous deformations of the lattice, in particular lattice merging and lattice splitting, common in geometric topology.

Universal quantum computing with lattice surgery
------------------------------------------------

To achieve universal quantum computing, we need to be able to perform all Clifford gates, and, in particular, CNOT gates.
Further, we need to be able to reliably inject states to enable `magic state injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
This is a bottom-up way to show that lattice surgery enables universal quantum computing and done so in its original introduction [#latticesurgery]_. 

Let us alternatively take a top-down approach here and show that we can perform arbitrary `Pauli product measurements <https://pennylane.ai/compilation/pauli-product-measurement>`__ (PPMs), 
because we know this enables universal quantum computing, as illustrated in, e.g., the :doc:`Game of Surface Codes <demos/gosc>` [#Litinski]_).
The gist of it is that `Pauli product rotations <https://pennylane.ai/compilation/pauli-product-rotations>`__ (PPRs) like :math:`e^{-i \tfrac{\theta}{2} P}` for arbitrary :doc:`Pauli words <~.pauli.PauliWord>` :math:`P`
with :math:`\theta \in \{k \tfrac{k}{4} | k \in \mathbb{Z} \}` cover the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ gate set.
These PPRs can either be directly executed on the lattice surgery based quantum computer using PPMs. Typically, this involves an auxiliary qubit in a specific state.
Clifford angles that are odd multiples of :math:`\frac{\pi}{2}` can be performed with an auxiliary qubit in :math:`|0\rangle` like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/clifford_PPM.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Non-clifford PPRs can be realized using a magic resource state like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/non_clifford_PPM.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Note that Pauli operations that have angles that are multiples of 
:math:`\pi` do not need to be executed on hardware, but can be tracked in software.
Further, we stress that there are different circuit identities to realize PPRs via PPMs, with the ones shown here just the basic examples taken from [#Litinski]_.
For our purposes here it suffices to note that realizing PPRs via PPMs is possible, so the only thing left to show is how to perform arbitrary PPMs using lattice surgery.







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
# .. [#braiding]
#
#     Robert Raussendorf, Jim Harrington, Kovid Goyal,
#     "Topological fault-tolerance in cluster state quantum computation",
#     `arXiv:quant-ph/0703143 <https://arxiv.org/abs/quant-ph/0703143>`__, 2007
#
#
# .. [#latticesurgery]
#
#     Dominic Horsman, Austin G. Fowler, Simon Devitt, Rodney Van Meter,
#     "Surface code quantum computing by lattice surgery",
#     `arXiv:1111.40226 <https://arxiv.org/abs/1111.4022>`__, 2011
#
#
# .. [#Fowler]
#
#     Austin G. Fowler, Craig Gidney
#     "Low overhead quantum computation using lattice surgery"
#     `arXiv:1808.06709 <https://arxiv.org/abs/1808.06709>`__, 2018.
#
#
# .. [#Litinski]
#
#     Daniel Litinski
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
#     `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.
#
#
# .. [#Chamberland]
#
#     Christopher Chamberland, Earl T. Campbell
#     "Universal quantum computing with twist-free and temporally encoded lattice surgery",
#     `arXiv:2109.02746 <https://arxiv.org/abs/2109.02746>`__, 2021
#
#
# Disclaimer
# ----------
# This demo is a Frankenstein of the two seminal papers on surface code quantum computing [#surfacecode]_ and [#latticesurgery]_.
# First two sections follow closely the intro sections of reference [#surfacecode]_, and the final section is inspired by [#latticesurgery]_.

