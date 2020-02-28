.. role:: html(raw)
   :format: html

.. _glossary_circuit_ansatz:

Circuit Ansatz
--------------

In the context of variational circuits, an *ansatz* usually describes a subroutine consisting of a sequence of gates
applied to specific wires. Similar to the architecture of a neural network, this only defines the base structure,
while the types of gates and/or their free parameters can be optimized by the variational procedure.

Many variational circuit ansaetze [#]_ have been proposed by the quantum computing community. The strength
of an ansatz depends on the desired use-case, and it is not always clear what makes a good ansatz.

One can distinguish three different base structures, namely a
**layered gate ansatz**, an **alternating operator ansatz**, and a **tensor network ansatz**.

Layered gate architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~

A *layer* is a sequence of gates that is repeated. The number of repetitions
of a layer forms a hyperparameter of the variational circuit.


We can often decompose a layer further into two overall unitaries :math:`A` and :math:`B`.

.. figure:: ../_static/concepts/vc_general.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Block :math:`A` contains single-wire gates applied to every subsystem or wire. Block :math:`B` consists of
both single-wire gates as well as entangling gates.


.. figure:: ../_static/concepts/vc_gatearchitecture.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Layered gate ansaetze can differ in three regards:

* Whether only :math:`A`, only :math:`B`, or both :math:`A` and :math:`B` are parametrized
* Which types of gates are used in :math:`A` and :math:`B`
* Whether the gates in Block :math:`B` are arranged randomly, fixed, or determined by a hyperparameter

Such layered ansaetze appear in both discrete and continuous-variable quantum computing models.

A parametrized, B fixed
***********************

In the simplest case of qubit-based devices, we can use general SU(2) gates (i.e., rotations) :math:`R` in
Block :math:`A` and let :math:`B` be fixed.


.. figure:: ../_static/concepts/vc_staticent.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


A parametrized, B parametrized
******************************

We can also have both :math:`A` and :math:`B` parametrized and the arrangements of the two-qubit gates
depends on a hyperparameter defining the range of two-qubit
gates (see also `Romero, Olson and Aspuru-Guzik (2016) <https://arxiv.org/abs/1612.02806>`_,
`Schuld et al. (2018) <https://arxiv.org/abs/1804.00633>`_).


.. figure:: ../_static/concepts/vc_cc.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


A fully parametrized architecture specific to continuous-variable systems has been proposed in
`Schuld & Killoran (2018) <https://arxiv.org/abs/1803.07128>`_.


.. figure:: ../_static/concepts/vc_cvkernels.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


The entangling layer :math:`B` contains an interferometer, a passive optical circuit made up of individual
beamsplitters and phase shifters. Block :math:`A` consists of single-mode gates with consecutively higher
order for the quadrature operator :math:`\hat{x}` which generates the gate: the displacement gate :math:`D`
is order-1, the quadratic phase gate :math:`Q` is order-2, and the cubic phase gate :math:`V` is order-3.

A fixed, B parametrized
***********************

An example where the single-qubit gates are fixed is a so-called *Instantaneous Quantum Polynomial (IQP)*
circuit, where :math:`A` consists of Hadamard gates and :math:`B` is made up of parametrized diagonal
one- and two-qubit gates (`Shepherd & Bremner (2008) <https://arxiv.org/abs/0809.0847>`_,
`Havlicek et al. (2018) <https://arxiv.org/abs/1804.11326>`_).


.. figure:: ../_static/concepts/vc_iqp.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Analogous circuits can also be considered for continuous-variable systems
`Arrazola, Rebentrost and Weedbrook (2017) <https://arxiv.org/abs/1712.07288>`_.


.. figure:: ../_static/concepts/vc_iqp_cv.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


IQP circuits are structured so that all gates in the :math:`B` block are diagonal in the computational basis.

Other structures
****************

Generalizing the simple two-block structure allows to build more complex layers, such as this layer of a
photonic neural network which emulates how information is processed in classical neural
nets (`Killoran et al. (2018) <https://arxiv.org/abs/1806.06871>`_,
`Steinbrecher et al. (2018) <https://arxiv.org/abs/1808.10047>`_).


.. figure:: ../_static/concepts/vc_cvqnn.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Alternating operator ansatz
~~~~~~~~~~~~~~~~~~~~~~~~~~~


The alternating operator ansatz was first introduced by
`Farhi, Goldstone and Gutmann (2014) <https://arxiv.org/abs/1411.4028>`_ as the
*Quantum Approximate Optimization Algorithm* (QAOA), and later used
for machine learning (`Verdon, Broughton, Biamonte (2017) <https://arxiv.org/abs/1712.05304>`_)
and other domain-specific applications (`Fingerhuth et al. (2018) <https://arxiv.org/abs/1810.13411>`_).

Again, we use layers of two blocks. The difference is that this time the unitaries representing
these blocks are defined via Hamiltonians :math:`A` and :math:`B` which are evolved for a short time :math:`\Delta t`.

.. figure:: ../_static/concepts/vc_aoa.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


The idea of this ansatz is based on analogies to adiabatic quantum computing, in which the system starts
in the ground state of :math:`A` and adiabatically evolves to the ground state of  :math:`B`. Quickly
alternating (i.e., *stroboscopic*) applications of  :math:`A` and  :math:`B` for very short times :math:`\Delta t`
can be used as a heuristic to approximate this evolution.

Tensor network ansatz
~~~~~~~~~~~~~~~~~~~~~

Amongst the architectures that do not consist of layers, but a single fixed structure, are gate sequences
inspired by tensor networks (`Huggins et al. (2018) <https://arxiv.org/abs/1803.11537>`_,
`Du et al. (2018) <https://arxiv.org/abs/1810.11922>`_). The simplest one is a tree
architecture that consecutively entangles subsets of qubits.


.. figure:: ../_static/concepts/vc_tree.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Another tensor network is based on *matrix product states*. The circuit unitaries can be decomposed in different ways,
and their size corresponds to the "bond dimension" of the matrix product state — the higher the bond dimension,
the more complex the circuit ansatz.


.. figure:: ../_static/concepts/vc_mps.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


.. note::

    Tensor networks such as matrix product states were invented to simulate certain quantum systems
    efficiently (though not universally) on classical computers. Hence, tensor network architectures do not
    necessarily give rise to classically intractable quantum nodes, but have found use as machine learning
    models (`Stoudenmire & Schwab (2016) <https://arxiv.org/abs/1605.05775>`_).

.. seealso:: In PennyLane, an ansatz is called a *template*. PennyLane contains
    a :ref:`growing library <intro_ref_temp>` of such circuit architectures.

.. rubric:: Footnotes

.. [#] "Ansaetze" is the German plural for "ansatz".
