r"""

Lattice Surgery
===============

Lattice surgery is a way to fault-tolerantly perform operations on two-dimensional platforms
with local physical connectivity. It enables lower spatial overheads for topological quantum
error correction codes such as surface codes or color codes.
In this demo, we are going to see how the basic operations, lattice merging and lattice splitting,
enable parity measurements of arbitrary Pauli operators, which unlock universal fault
tolerant quantum computing.

.. figure:: ../_static/demonstration_assets/block_encoding/thumbnail_Block_Encodings_Matrix_Oracle.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Surface code quantum computing
------------------------------

Topological quantum error correction codes like the surface code [#surfacecode]_ protect quantum information of logical qubits by
ensuring the underlying physical qubits are kept in the code space, defined by the :math:`+1` eigenspace of a set of stabilizer measurements.
A logical qubit is then represented by a patch of physical qubits, called data qubits, on a square lattice like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit1.png
    :align: center
    :width: 25%
    :target: javascript:void(0)
  
Each intersection of a line corresponds to a data qubit. Here we have a code distance of :math:`d=5`, so :math:`5 \times 5` data qubits in total. 
On yellow surfaces, we continually measure X stabilizers, which simply are the :math:`X_a X_b X_c X_d` operators
on the four data qubits of the corners. Similarly, Z stabilizers are measured on white surfaces. 
These stabilizer measurements are not performed directly on the data qubits, but via another kind of qubit that sits at the center of each square surface
as well as the center of each arch (not shown here, see below). 
These extra qubits are sometimes called measurement qubits or syndrome qubits and the measurement is performed by entangling the data qubits with the syndrome qubit and measuring it in the corresponding basis (see Fig. 1 in [#surfacecode]_).

Note that this representation of a qubit is in the so-called rotated surface code due to its :math:`45^\circ` rotation with respect to the original planar surface code.
Here we indicate the underlying data and syndrome qubits with black and red dots, respectively. 
The vertical lines do not correspond to physical connections, but are a mere guide to the eye that differentiate X vertex and Z face syndromes. In principle, all
nearest neighbor qubits are physically connected and can perform noisy (Clifford + T) gates.

.. figure:: ../_static/demonstration_assets/lattice_surgery/surface_code_qubit2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

One such :math:`d \times d` patch represents a logical qubit. An important feature is that it has two X and two Z edges, as this enables the error-corrected encoding of the qubit (more on logical operators below).
We are interested in performing `logical` operations like a CNOT gate between two such patches that represent a logical qubit, each.
The most straight-forward way to compute a logical CNOT is by performing physical CNOT gates between the data qubits
of each logical qubit patch, indicated here by the red line exemplarily for one pair:

.. figure:: ../_static/demonstration_assets/lattice_surgery/transversal_cnot.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is called a transversal operation, which is problematic as it requires non-local physical connections between data qubits.
Most quantum hardware architectures only allow for nearest neighbor interactions, and thus do not allow for transversal gate operations.
Instead, we want to perform CNOT gates non-transversally. In the early days, this was achieved via braiding [#braiding]_, 
a concept commonly encountered in algebraic topology.
In this setting, qubits are encoded by defects in the code, and operations via continuous deformations of the code.
However, defect based qubits suffer from requiring significantly more physical qubits per logical qubit.

This is where lattice surgery [#latticesurgery]_ comes into play, as it has been shown to enable error-corrected logical operations
with significantly lower space resources with comparable time requirements [#Fowler]_ [#Litinski]_ [#Chamberland]_. The fundamental operations in lattice
surgery are discontinuous deformations of the lattice, in particular lattice merging and lattice splitting, common in geometric topology.

Universal quantum computing with Pauli product measurements
-----------------------------------------------------------

To achieve universal quantum computing, we need to be able to perform all Clifford gates, and, in particular, CNOT gates.
Further, we need to be able to reliably inject states to enable `magic state injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
This is a bottom-up way to show that lattice surgery enables universal quantum computing and done so in its original introduction [#latticesurgery]_. 

Let us alternatively take a top-down approach here and show that we can perform arbitrary `Pauli product measurements <https://pennylane.ai/compilation/pauli-product-measurement>`__ (PPMs), 
because we know this enables universal quantum computing, as illustrated in, e.g., the :doc:`Game of Surface Codes <demos/tutorial_gosc>` [#Litinski]_).
The gist of it is that `Pauli product rotations <https://pennylane.ai/compilation/pauli-product-rotations>`__ (PPRs) like :math:`e^{-i \tfrac{\theta}{2} P}`, represented as

.. figure:: ../_static/demonstration_assets/lattice_surgery/PPR.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

for arbitrary :class:`Pauli words <~.pauli.PauliWord>` :math:`P`
with Clifford and non-Clifford angles :math:`\theta \in \{k \tfrac{\pi}{4} | k \in \mathbb{Z} \}` cover the `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ gate set, and thus allow for universal quantum computing.
These PPRs can be directly executed using PPMs in the following way, so all we need to show is that the lattice surgery based quantum computer can perform arbitrary PPMs. Typically, this involves an auxiliary qubit in a specific state.
Clifford angles that are odd multiples of :math:`\frac{\pi}{2}` can be performed with an auxiliary qubit in :math:`|0\rangle` like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/clifford_PPM.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

We use the color coding from [#Litinski]_, where orange boxes correspond to Clifford gates, 
gray boxes to Pauli operators (as :math:`e^{-i \tfrac{(2k+1)\pi}{2}P} \propto P`), and green boxes to non-Clifford gates.
Non-clifford PPRs can be realized using a magic resource state like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/non_clifford_PPM.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Here, we injected a magic state :math:`T|+\rangle = \tfrac{1}{\sqrt{2}} \left(|0\rangle + e^{-i \tfrac{1}{4}} |1\rangle\right)` that was produced separately via `magic state distillation <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`__.
Note that Pauli operations that have angles that are multiples of 
:math:`\pi` do not need to be executed on hardware, but can be tracked in software.
Further, we stress that there are different circuit identities to realize PPRs via PPMs, with the ones shown here just the basic examples taken from [#Litinski]_.
For our purposes here it suffices to note that realizing PPRs via PPMs is possible, so the only thing left to show is how to perform arbitrary PPMs using lattice surgery.

Single qubit measurements
^^^^^^^^^^^^^^^^^^^^^^^^^

Before we get to the meat of performing arbitrary PPMs with lattice surgery, let us recal how to perform single qubit measurements (and thus single qubit gates).
Logical operators :math:`Z_L` and :math:`X_L` of the logical qubit are defined by a horizontal line of :math:`Z` and vertical line of :math:`X` measurements
on the data qubits like so:

.. figure:: ../_static/demonstration_assets/lattice_surgery/logical_X_Z.png
    :align: center
    :width: 30%
    :target: javascript:void(0)

Note that these are not unique, but rather examples of the topological equivalence class of connected lines between the two Z and X edges, respectively.
Any such operator represents the logical operators :math:`Z_L` and :math:`X_L`. Because we only care about the homology of the measurement
(i.e., that it connects the two kinds of edges), which is why this is also called *homological measurement*.
It does not really matter which operator of the equivalence class we measure, and in principle we have access to any of them without extra effort because they are related to each other via stabilizer measurements, 
which we anyway perform during the continually performed error correction cycles.
So for the following, we will just consider those that are convenient for illustrational purposes.

An important feature of the logical operators is that they are not stabilizers, but commute with all stabilizers to ensure that the logical operator does not move the 
state out of the code space (:math:`O_L S |\psi\rangle_L = S O_L |\psi\rangle_L`).
The commutation can be seen from the fact
that the logical operator only ever overlaps with an even multiple of X or Z operators, and thus commutes with any other stabilizer.
Stabilizers are topologically trivial (loops), whereas logical operators are topologically non-trivial (cycles). This differentiation is only possible due to
the distinctive Z and X boundaries, which enable the definition of the logical operators without being stabilizers themselves.

We also note that logical :math:`Y_L` measurement is a topic on its own. In principle, one could measure :math:`Y_L` by simultaneously measuring the logical
:math:`X_L` and :math:`Z_L` with a physical :math:`Y` measurement on the intersecting data qubit.
However, this is not a properly defined logical operator anymore as it does not commute with the stabilizers, and thus moves the qubit out of the code space.
Measuring (and applying, same same really) logical :math:`Y_L` operators is still possible, just a little more complicated as we will show further below.

Arbitrary Pauli product measurements via lattice merging and splitting
----------------------------------------------------------------------

We now want to show how to perform the two fundamental operations of measuring :math:`Z_L \otimes Z_L` and :math:`X_L \otimes X_L` between two surface code qubits via lattice surgery.
The operation is fairly simple in principle: merge the two patches, and split them again. We just need to make sure that the qubits are facing each other with the correct edges,
and use the correct state preparation and measurements for the merging and splitting operations, respectively.

Let us walk through the process of measuring :math:`X_L \otimes X_L` first. 
Assuming we already have the two qubits facing each other with their smooth Z edges, 
we can start by preparing the intermediate data qubits in :math:`|0\rangle`.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Note that here we left just a single column of data qubits between the two patches, but in principle we can have a larger area, as long as we can make the boundaries match.
The actual merging is then done by simply by including the intermediate X and Z stabilizers in the error correction cycles.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Conveniently, the product of all stabilizers between the two logical :math:`X_L` operators, indicated by the red dots below, corresponds to the eigenvalue of :math:`X_L \otimes X_L`.
In case any of the two :math:`X_L` operators were carrying a sign, we need to make sure to include those in the product.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX3.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Now that we retrieved our measurement result :math:`X_L \otimes X_L`, we want to restore the two qubits, which is achieved by lattice splitting. This, on the other hand,
is done by measuring the intermediate data qubits in the Z basis.

.. figure:: ../_static/demonstration_assets/lattice_surgery/XX4.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

In case the green Z measurement in the middle yields a negative sign, 
we need to assign it to one of the two logical :math:`Z_L` operators, 
on top of the product of the sign of the original two :math:`Z_L` operators we originally started from.

Measuring :math:`Z_L \otimes Z_L` works in the same fashion, but with reversed roles: We face rough X edges towards each other, initialize in :math:`|+\rangle` to merge and measure in :math:`X` to split again:

.. figure:: ../_static/demonstration_assets/lattice_surgery/ZZ.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

:math:`Y_L` measurements
------------------------

At this point we can measure arbitrary :math:`X_L`- and :math:`Z_L`- Pauli product measurements. The last missing ingredient for universality is measuring :math:`Y_L` operators.

Introducing discontinuous operations on the surface code does not exclude that we can still use the continuous transformations of the code.
In particular, we can always extend a qubit to a larger surface or move the edges of it. These are important operations if we want to include logical :math:`Y_L` measurements.

The operations are very similar to the merging and splitting operations. To extend a qubit patch on the smooth Z edge, we initialize the neighboring data qubits in :math:`|0\rangle` 

.. figure:: ../_static/demonstration_assets/lattice_surgery/extend1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

and perform :math:`d` cycles of error correction:

.. figure:: ../_static/demonstration_assets/lattice_surgery/extend2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

To remove part of the extended patch again, we perform :math:`Z` measurements on the data qubits, effectively moving the qubit one patch size to the right:

.. figure:: ../_static/demonstration_assets/lattice_surgery/extend3.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

(Recall from earlier that green and red lines merely indicate representatives of the logical :math:`Z_L` and :math:`X_L` operators, so moving the red line to the right is free with the error correction cycles)

Extending the qubit on the rough :math:`X` edge works similarly, but moving vertically, initializing in :math:`|+\rangle` and measuing :math:`X` on the data qubits.

We can also move (or rotate?) the type of the edges within :math:`d` error correction code cycles. This makes most sense on an already extended qubit because this way we do not change the code distance.

.. figure:: ../_static/demonstration_assets/lattice_surgery/corner_moving.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
  
Here we show an extended single qubit patch and three example re-orientations of the type of edges.
The smaller images are guides to the eye to indicate the settings, with X edges as solid lines and Z edges as dotted lines (i.e. the same as the logical measurements moved to the edges).
Note that the important property is for neighboring X and Z edges to overlap on two data qubits for the corresponding stabilizers to commute.

Finally, to measure $Y$ we perform the following procedure:

.. figure:: ../_static/demonstration_assets/lattice_surgery/y_measurement.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

Here, one extends and orients a qubit such that it has both :math:`X` and :math:`Z` edges on the same side. 
At the same time, an auxiliary qubit is initialized in :math:`|0\rangle` parallel to our extended qubit.
This is intentionally done such that there is a mismatch between the boundary stabilizers which introduces a so-called twist defect, highlighted in purple.
When we perform lattice merging the two qubits, we obtain new stabilizers, some of which are mixed Z and X!
This is fine because they still commute with all other stabilizers. This can be shown by checking that :math:`[ZZ, XY] = [YZ,XX] = [XZ, ZX] = 0`.

Some of these are trivial in the sense that they are simply the product of the already-measured stabilizers from each of the qubits. 
The non-trivial new ones are highlighted in red and correspond to the measurement of :math:`Y \otimes Z` of the joint state :math:`|\psi\rangle \otimes |0\rangle`
between the qubit and the auxiliary qubit in :math:`|0\rangle`, effectively measuring :math:`\langle Y \rangle = \langle \psi | Y |\psi \rangle = \langle \psi 0 | Y \otimes Z |\psi 0 \rangle`.
A more intuitive way to view this is that we simultaneously measured the X and Z edge of our extended qubit, yielding the :math:`Y \propto X Z` measurement (modulo some global phase). This is exactly the perspective in the game of surface codes [#Litinski]_.

Measuring :math:`Y_L` inside a Pauli product measurement, e.g. :math:`Y_L \otimes Z_L` works in the same fashion
as above with lattice merging and splitting while making sure a twist defect is introduced. This corresponds to having
the first qubit facing the other with both X and Z edges.

More details on twist-based lattice surgery can be found in [#Litinski2]_.

"""


##############################################################################
# 
#
# Conclusion
# ----------
#
# Lattice surgery is the answer to performing error-corrected quantum computing on planar codes
# with local connectivity as it allows for non-transversal gates. We introduced its basics
# exemplified on the surface code and by realizing arbitrary Pauli product measurements.
# These are performed by merging and splitting qubits in such a way that the logical operators "topologically" align.
# In this process of merging and splitting, we only ever change which stabilizers we measure, but we never perform a strict projective measurement of the logical operator.
# In a sense, we are reading out parity information from the stabilizer measurements in a non-pertubative way.
# The parity on the other hand provides us only with the topological information of what the logical string is connecting,
# which is why this process is called homological measurement.
#
# Note that the introduction of the higher weight twist-defect stabilizer with 5 Pauli operators can be problematic,
# which is why twist-free alternatives are also being proposed [#Chamberland]_.
# 
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
# .. [#Litinski2]
#
#     Daniel Litinski, Felix von Oppen
#     "Lattice Surgery with a Twist: Simplifying Clifford Gates of Surface Codes",
#     `arXiv:1709.02318 <https://arxiv.org/abs/1709.02318>`__, 2017
#
#
# Disclaimer
# ----------
# This demo is a Frankenstein of the two seminal papers on surface code quantum computing [#surfacecode]_ and [#latticesurgery]_.
# First two sections follow closely the intro sections of reference [#surfacecode]_, and the final section is inspired by [#latticesurgery]_.

