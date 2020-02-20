.. role:: html(raw)
   :format: html

.. _glossary_circuit_ansatz:

Circuit Ansatz
--------------

In the context of variational circuits, an *ansatz* usually describes a subroutine consisting of a sequence of gates
applied to specific wires. Similar to the architecture of a neural network, this only defines the base structure,
while the types of gates and/or their free parameters can be optimized by the variational procedure.

Many variational circuit architectures have been proposed by the quantum computing community [#]_. The strength
of an architecture varies depending on the desired use-case, and it is not always clear what makes a good ansatz.

One can distinguish three different base structures of architectures, namely
**layered gate architectures**, **alternating operator architectures** and **tensor network architectures**.

.. seealso:: In PennyLane, an ansatz is called a *template*. PennyLane contains
    a :ref:`growing library <intro_ref_temp>` of such circuit architectures.

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


Layered gate architectures can differ in three regards:

* Whether only :math:`A`, only :math:`B`, or both :math:`A` and :math:`B` are parametrized
* Which types of gates are used in :math:`A` and :math:`B`
* Whether the gates in Block :math:`B` are arranged randomly, fixed, or determined by a hyperparameter

Such layered architectures appear in both discrete and continuous-variable quantum computing models.

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
gates (see also :cite:`romero2017quantum`, :cite:`schuld2018circuit`).


.. figure:: ../_static/concepts/vc_cc.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


A fully parametrized architecture specific to continuous-variable systems has been proposed in :cite:`schuld2018quantum`.


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
one- and two-qubit gates :cite:`shepherd2009temporally`:cite:`havlicek2018supervised`.


.. figure:: ../_static/concepts/vc_iqp.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Analogous circuits can also be considered for continuous-variable systems :cite:`arrazola2017quantum`.


.. figure:: ../_static/concepts/vc_iqp_cv.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


IQP circuits are structured so that all gates in the :math:`B` block are diagonal in the computational basis.

Other structures
****************

Generalizing the simple two-block structure allows to build more complex layers, such as this layer of a
photonic neural network which emulates how information is processed in classical neural
nets :cite:`killoran2018continuous` :cite:`steinbrecher2018quantum`.


.. figure:: ../_static/concepts/vc_cvqnn.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Alternating operator architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The alternating operator architecture was first introduced in Farhi and Goldstone's
*Quantum Approximate Optimization Algorithm* (QAOA) :cite:`farhi2014quantum` and later used
for machine learning :cite:`verdon2017quantum` and other domain-specific applications :cite:`fingerhuth2018quantum`.

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

Tensor network architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amongst the architectures that do not consist of layers, but a single fixed structure, are gate sequences
inspired by tensor networks :cite:`huggins2018towards` :cite:`du2018expressive`. The simplest one is a tree
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
    models :cite:`miles2016supervised`.


.. rubric:: Footnotes

.. [#] For example, see the following non-exhaustive list: :cite:`shepherd2009temporally`
    :cite:`farhi2014quantum` :cite:`miles2016supervised` :cite:`romero2017quantum` :cite:`arrazola2017quantum`
    :cite:`farhi2017quantum` :cite:`benedetti2018generative` :cite:`huggins2018towards` :cite:`schuld2018quantum`
    :cite:`havlicek2018supervised` :cite:`schuld2018circuit` :cite:`dallaire2018quantum` :cite:`killoran2018continuous`
    :cite:`steinbrecher2018quantum`.
