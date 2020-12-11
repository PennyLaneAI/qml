r"""
Measurement optimization
========================

.. meta::
    :property="og:description": Optimize and reduce the number of measurements required to evaluate a variational algorithm cost function.
    :property="og:image": https://pennylane.ai/qml/_images/sphx_glr_tutorial_rosalin_002.png

.. related::

   tutorial_vqe Variational quantum eigensolver
   tutorial_quantum_chemistry Quantum chemistry with PennyLane
   tutorial_qaoa_intro Intro to QAOA

The variational quantum eigensolver (VQE) is the OG variational quantum algorithm. Harnessing
near-term quantum hardware to solve for the electronic structure of molecules, VQE is *the*
algorithm that sparked the variational circuit craze of the last 5 years, and holds the greatest
promise for showcasing a quantum advantage on near-term quantum hardware. It has also inspired
other quantum algorithms such as the :doc:`Quantum Approximate Optimization Algorithm (QAOA)
</demos/tutorial_qaoa_intro>`.

To scale VQE beyond the regime of classical computation, however, we need to use it to solve for the
ground state of excessively larger and larger molecules. A side effect is that the number of
measurements we need to make on the quantum hardware also grows polynomially---a huge bottleneck,
especially when quantum hardware access is limited and expensive.

To mitigate this 'measurement problem', a plethora of recent research dropped over the course
of 2019 and 2020, exploring potential strategies to minimize the number of measurements required.
In fact, by grouping qubit-wise commuting terms of the Hamiltonian, we can significantly reduce the
number of measurements needed---in some cases, reducing the number of measurements by up to
90%(!).

In this demonstration, we revisit VQE, see first-hand how the required number of measurements scales
as molecule size increases, and finally use these measurement optimization strategies
to minimize the number of measurements we need to make.

It all begins with VQE
----------------------

The study of :doc:`variational quantum algorithms </glossary/variational_circuit>` was spearheaded
by the introduction of the :doc:`variational quantum eigensolver <tutorial_vqe>` (VQE) algorithm in
2014 [#peruzzo2014]_. While classical variational techniques have been known for decades to estimate
the ground state energy of a molecule, VQE allowed this variational technique to be applied using
quantum computers. Since then, the field of variational quantum algorithms has evolved
significantly, with larger and more complex models being proposed (such as
:doc:`quantum neural networks </demos/quantum_neural_net>`, :doc:`QGANs </demos/tutorial_QGAN>`, and
:doc:`variational classifiers </demos/tutorial_variational_classifier>`). However, quantum chemistry
remains one of the flagship uses-cases for variational quantum algorithms, and VQE the standard-bearer.

The appeal of VQE lies almost within its simplicity. A circuit ansatz :math:`U(\theta)` is chosen
(typically the Unitary Coupled-Cluster Singles and Doubles
(:func:`~pennylane.templates.subroutines.UCCSD`) ansatz), and the qubit representation of the
molecular Hamiltonian is computed:

.. math:: H = \sum_i c_i h_i,

where :math:`h_i` are the terms of the Hamiltonian written as a product of Pauli operators :math:`\sigma_n`:

.. math:: h_i = \prod_{n=0}^{N} \sigma_n.

The cost function of VQE is then simply the expectation value of this Hamiltonian after
the variational quantum circuit:

.. math:: \text{cost}(\theta) = \langle 0 | U(\theta)^\dagger H U(\theta) | 0 \rangle.

By using a classical optimizer to *minimize* this quantity, we will be able to estimate
the ground state energy of the Hamiltonian :math:`H`:

.. math:: H U(\theta_{min}) |0\rangle = E_{min} U(\theta_{min}) |0\rangle.

In practice, when we are using quantum hardware to compute these expectation values we expand out
the summation, resulting in separate expectation values that need to be calculated for each term in
the Hamiltonian:

.. math::

    \text{cost}(\theta) = \langle 0 | U(\theta)^\dagger \left(\sum_i c_i h_i\right) U(\theta) | 0 \rangle
                        = \sum_i c_i \langle 0 | U(\theta)^\dagger h_i U(\theta) | 0 \rangle.

.. note::

    How do we compute the qubit representation of the molecular Hamiltonian? This is a more
    complicated story, that involves applying a self-consistent field method (such as Hartree-Fock),
    and then performing a fermionic-to-qubit mapping such as the Jordan-Wigner or Bravyi-Kitaev
    transformations.

    For more details on this process, check out the :doc:`/demos/tutorial_quantum_chemistry`
    tutorial.

The measurement problem
-----------------------

For small molecules, VQE scales and performs exceedingly well. For example, for the
Hydrogen molecule :math:`\text{H}_2`, the final qubit-representation Hamiltonian
has 15 terms that need to be measured. Lets generate this Hamiltonian from the electronic
structure file :download:`h2.xyz </demonstrations/h2.xyz>`, using PennyLane
QChem to verify the number of terms.
"""

import functools
from pennylane import numpy as np
import pennylane as qml

H, num_qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

print("Required number of qubits:", num_qubits)
print(H)

###############################################################################
# Here, we can see that the Hamiltonian involves 15 terms, so we expect to compute 15 expectation values
# on hardware. Let's generate the cost function to check this.

# Create a 4 qubit simulator
dev = qml.device("default.qubit", wires=num_qubits)

# number of electrons
electrons = 2

# Define the Hartree-Fock initial state for our variational circuit
initial_state = qml.qchem.hf_state(electrons, num_qubits)

# Construct the UCCSD ansatz
singles, doubles = qml.qchem.excitations(electrons, num_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
ansatz = functools.partial(
    qml.templates.UCCSD, init_state=initial_state, s_wires=s_wires, d_wires=d_wires
)

# generate the cost function
cost = qml.ExpvalCost(ansatz, H, dev)

##############################################################################
# If we evaluate this cost function, we can see that it corresponds to 15 different
# QNodes under the hood---one per expectation value:

params = np.random.normal(0, np.pi, len(singles) + len(doubles))
print("Cost function value:", cost(params))
print("Number of quantum evaluations:", dev.num_executions)

##############################################################################
# How about a larger molecule? Lets try the water molecule :download:`h2o.xyz </demonstrations/h2o.xyz>`:

H, num_qubits = qml.qchem.molecular_hamiltonian("h2o", "h2o.xyz")

print("Required number of qubits:", num_qubits)
print("Number of Hamiltonian terms/required measurements:", len(H.ops))

print("\n", H)


##############################################################################
# Simply going from two atoms in :math:`\text{H}_2` to three in :math:`\text{H}_2 \text{O}`
# resulted in 2050 measurements that must be made!
#
# We can see that as the size of our molecule increases, we run into a problem; larger molecules
# result in Hamiltonians that not only require a larger number of qubits :math:`N` in their
# representation, but the number of terms in the Hamiltonian scales like
# :math:`\mathcal{O}(N^4)`! 😱😱😱
#
# We can mitigate this somewhat by choosing smaller `basis sets
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`__ to represent the electronic structure
# wavefunction, however this comes with an accuracy cost, and doesn't reduce the number of
# measurements significantly enough to allow us to scale to classically intractable problems.
#
# .. figure:: /demonstrations/measurement_optimize/n4.png
#     :width: 70%
#     :align: center
#
#     The number of qubit Hamiltonian terms required to represent various molecules in the specified
#     basis sets (adapted from `Minimizing State Preparations for VQE by Gokhale et al.
#     <https://pranavgokhale.com/static/Minimizing_State_Preparations_for_VQE.pdf>`__)


##############################################################################
# Grouping qubit-wise commuting terms
# -----------------------------------

##############################################################################
# Beyond VQE
# ----------

##############################################################################
# References
# ----------
#
# .. [#peruzzo2014]
#
#     Alberto Peruzzo, Jarrod McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [yen2020]
#
#     Tzu-Ching Yen, Vladyslav Verteletskyi, and Artur F. Izmaylov. "Measuring all compatible
#     operators in one series of single-qubit measurements using unitary transformations." `Journal of
#     Chemical Theory and Computation 16.4 (2020): 2400-2409.
#     <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00008>`__
#
# .. [verteletskyi2020]
#
#     Vladyslav Verteletskyi, Tzu-Ching Yen, and Artur F. Izmaylov. "Measurement optimization in the
#     variational quantum eigensolver using a minimum clique cover." `The Journal of Chemical Physics
#     152.12 (2020): 124114. <https://aip.scitation.org/doi/10.1063/1.5141458>`__
#
# .. [izmaylov2019]
#
#    Artur F. Izmaylov, et al. "Unitary partitioning approach to the measurement problem in the
#    variational quantum eigensolver method." `Journal of Chemical Theory and Computation 16.1 (2019):
#    190-195. <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00791>`__
#
# .. [gokhale2020]
#
#    Pranav Gokhale, et al. "Minimizing state preparations in variational quantum eigensolver by
#    partitioning into commuting families." `arXiv preprint arXiv:1907.13623 (2019).
#    <https://arxiv.org/abs/1907.13623>`__
