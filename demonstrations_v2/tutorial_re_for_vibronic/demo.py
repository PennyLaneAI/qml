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

In this demo, we will use PennyLane's resource estimation tools to "stress test" a vibronic dynamics algorithm. We will calculate the precise logical resource requirements,
allowing us to optimize our strategy before hardware becomes available.

.. figure:: ../_static/demonstration_assets/mapping/long_image.png
    :align: center
    :width: 80%
    :target: javascript:void(0)


The Vibronic Hamiltonian
------------------------

To perform this feasibility check, we must first define the system. We use the **Vibronic Coupling Hamiltonian**,
which describes a set of :math:`A` electronic states interacting with :math:`N` vibrational modes.

Unlike standard electronic structure problems, this model mixes discrete electronic levels with continuous
vibrational motion (bosons). The Hamiltonian takes the form:

.. math::

    H = T + V(\mathbf{\hat{q}})


For our resource estimation, the complexity is dictated by the key parameters found in
PennyLane's `~.pennylane.estimator.compact_hamiltonian.VibronicHamiltonian` class:

* ``num_modes``: The number of vibrational modes considered.
* ``num_states``: The number of electronic states.
* ``taylor_degree``: The order of the coupling expansion.
* ``grid_size``: The resolution of the position/momentum operators.
"""
#
# References
# ----------
#
# .. [#Tranter]
#
#      A. Tranter, S. Sofia, *et al.*, "The Bravyi–Kitaev Transformation:
#      Properties and Applications". `International Journal of Quantum Chemistry 115.19 (2015).
#      <https://onlinelibrary.wiley.com/doi/10.1002/qua.24969>`__
#
# .. [#Yordanov]
#
#      Y. S. Yordanov, *et al.*, "Efficient quantum circuits for quantum computational chemistry".
#      `Physical Review A 102.6 (2020).
#      <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.062612>`__
#
