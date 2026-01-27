r"""
Quantifying Resource Requirements for Vibronic Dynamics Simulation
==================================================================

Standard quantum chemistry often relies on the adiabatic Born-Oppenheimer approximation, assuming nuclei move on a single potential energy surface (PES)
well-separated from others. But in the world of photochemistry, where molecules absorb light, and break bonds, this approximation
breaks down. The timescales of electronic and nuclear motion become comparable, giving rise to vibronic coupling. This coupling, drives
critical processes like photosynthesis, and solar cell efficiency and accurately simulating this requires a Hamiltonian that treats the dynamics
of electrons and nuclei simultaneously.

However, modeling such dynamics is computationally demanding. Before we attempt to run such complex algorithms on future fault-tolerant hardware,
we need to answer a fundamental question: Is this algorithm actually feasible?

In this demo, we assume the role of an algorithm architect. We will construct a simulation workflow
for the vibronic Hamiltonian from the ground up, utilizing the modular building blocks within PennyLane's :mod:`~.pennylane.estimator`.
By constructing this pipeline ourselves, we can perform a feasibility analysis, determining exactly what resources are required to simulate
these complex non-adiabatic processes on future hardware.

.. figure:: ../_static/demonstration_assets/mapping/long_image.png
    :align: center
    :width: 80%
    :target: javascript:void(0)


The Vibronic Hamiltonian
------------------------
To perform this feasibility check, we must first define the system. We use the vibronic coupling Hamiltonian,
which describes a set of :math:`N` electronic states interacting with :math:`M` vibrational modes.

Unlike standard electronic structure problems, this model mixes discrete electronic levels with continuous
vibrational motion, and the Hamiltonian takes the form:

.. math::

    H = T + V,

where :math:`T` represents the kinetic energy of the nuclei, and is diagonal while :math:`V` contains off-diagonal
electronic couplings.

To simulate the time evolution :math:`U(t) = e^{-iHt}`, we employ a product formula (Trotterization) based approach as outlined
in D. Motlagh et al. [#Motlagh2025]. A general challenge in Trotter-based approaches is to identify an optimal
decomposition of the Hamiltonian into fragments. Here, we utilize a term-based fragmentation scheme, where the Hamiltonian terms
are grouped based on their vibrational monomial, e.g., collecting all electronic terms coupled to :math:`\hat{q}_1 \hat{q}_2`
in one fragment.

For each resulting fragment, the electronic component is a small, arbitrary Hermitian matrix. We apply a
unitary transformation to rotate this component into its eigenbasis. With this diagonalization, the associated
vibrational operator simplifies to a **single monomial**. This structure is key to the algorithm's feasibility,
as simulating a single monomial is significantly more efficient than simulating the complex polynomial sums
found in naive decompositions.

Defining the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^
With the algorithmic approach defined, the next step is to instantiate the system we wish to benchmark.

While access to the full Hamiltonian coefficients would allow us to further optimize costs by leveraging the commutativity of electronic parts,
we can still derive a reliable baseline using only structural parameters. By defining the number of modes, states, and coupling order,
we can construct a :class:`~.pennylane.estimator.VibronicHamiltonian` that mimics the cost topology of a real system without needing full
integral data.

Let's take the example of Anthracene dimer, a system critical for understanding singlet fission in organic solar cells [#Motlagh2025]_.
"""

from pennylane import estimator as qre
anthracene_ham = qre.VibronicHamiltonian(num_states=6, num_modes=21, grid_size=4, taylor_degree=2)

#####################################################################
# Constructing Circuits for one Time-Step
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Based on the term-based fragmentation scheme, the single Trotter step is composed of two distinct types of
# quantum circuits interleaved together:
#
# 1.  **Potential Energy Fragments:** These implement the interaction terms. For each fragment, the algorithm
#     loads coefficients using a `QROM (Quantum Read-Only Memory) <>`_, computes the vibrational monomial product
#     using quantum arithmetic, and applies a phase gradient.
# 2.  **Kinetic Energy Fragments:** These implement the nuclear kinetic energy. Since the kinetic operator depends
#     on momentum :math:`P`, this circuit uses the `Quantum Fourier Transform (QFT) <>`_ to switch to the momentum basis,
#     applies a phase rotation, and then switches back.
#

######################################################################
# With the Hamiltonian defined, we can move to defining the rest of the parameters needed for Trotterization, that is the number of Trotter
# steps and order of the Suzuki-Trotter expansion. For the sake of brevity, we take these numbers directly from the reference. [#Motlagh2025]_


#
# References
# ----------
#
# .. [#Motlagh2025]
#
#      Danial Motlagh et al., "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design".
#      `arXiv preprint arXiv:2411.13669 (2014) <https://arxiv.org/abs/2411.13669>`__.
#
#