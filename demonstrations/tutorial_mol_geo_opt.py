r"""
Optimization of molecular geometries
====================================

.. meta::
    :property="og:description": Find the equilibrium geometry of a molecule
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_spectra_h2.png

.. related::
   tutorial_vqe Variational Quantum Eigensolver
   
*Author: PennyLane dev team. Last updated: 12 May 2021.*

Predicting the most stable arrangement of the atoms that conform a molecule is one of the most
important tasks in computational chemistry. This corresponds to an optimization problem where the
the total energy of the molecule has to be minimized with respect to the positions of the
atomic nuclei. Within the
`Born-Oppenheimer approximation <https://en.wikipedia.org/wiki/
Born%E2%80%93Oppenheimer_approximation>`_ [#kohanoff2006]_ the total electronic energy of the 
molecule :math:`E(x)` depends parametrically on the nuclear coordinates :math:`x` which defines
the potential energy surface. Solving the stationary problem :math:`\nabla_x E(x) = 0` corresponds
to what is known as *molecular geometry optimization* and the optimized nuclear coordinates
determine the *equilibrium geometry* of the molecule. For example, the figure below illustrates
these concepts for the `trihydrogen cation <https://en.wikipedia.org/wiki/Trihydrogen_cation>`_
molecule. Its equilibrium geometry in the electronic ground-state resembles an equilateral
triangle whose side length is the optimized H-H bond length :math:`d`.

|

.. figure:: /demonstrations/mol_geo_opt/fig_pes.png
    :width: 50%
    :align: center

|

Classical algorithms for molecular geometry optimization are computationally expensive. They
typically rely on the Newton-Raphson method [#jensenbook]_ requiring access to the nuclear
gradients and the Hessian of the energy at each optimization step while searching for the global
minimum along the potential energy surface :math:`E(x)`. As a consequence, using accurate
wave function methods to solve the molecule's electronic structure at each step is computationally
intractable even for medium-size molecules. Instead, density functional theory methods
[#dft_book_ are used to obtain approximated geometries.

Variational quantum algorithms for quantum chemistry applications use quantum computer to prepare
the electronic wave function of a molecule and to measure the expectation value of the Hamiltonian
while a classical optimizer adjusts the circuit parameters in order to minimize the total
energy [#mcardle2020]_. The problem of finding the equilibrium geometry of a molecule can be 
recast in terms of a more general variational quantum algorithm where the target electronic 
Hamiltonian :math:`H(x)` is a *parametrized* observable that depends on the nuclear
coordinates :math:`x`. This implies that the objective function, defined by the expectation value
of the Hamiltonian :math:`H(x)` computed in the trial state :math:`\vert \Psi(\theta) \rangle` prepared by a parametrized quantum circuit :math:`U(\theta)`, depends on both the circuit and the 
Hamiltonian parameters. Furthermore, the cost function can be minimized using a *joint* 
optimization scheme where the analytical gradients of the cost function with respect to circuit
and Hamiltonian parameters are computed simultaneously at each optimization step.
Note that this approach does not require nested optimizations of the circuit parameters for each 
set of nuclear coordinates, as occurs in the analogous classical algorithms. The optimized circuit
parameters determine the energy of the electronic state prepared by the quantum circuit, and the 
final set of nuclear coordinates is precisely the equilibrium geometry of the molecule in this
electronic state.

The variational quantum algorithm proceeds as follows:

#. Define the molecule for which we want to find the equilibrium geometry.

#. Build the parametrized electronic Hamiltonian :math:`H(x)` for a given set of
   nuclear coordinates.

#. Design the variational quantum circuit preparing the electronic state of the
   molecule :math:`\vert \Psi(\theta) \rangle`.

#. Define the cost function :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert
   \Psi(\theta) \rangle`.

#. Set the initial values for the circuit parameters :math:`\theta` and the
   nuclear coordinates :math:`x`.

#. Solve the optimization problem :math:`E = \min_{\{\theta, x\}} g(\theta, x)` using a
   gradient-descent optimizer to minimize the total energy of the molecule and to find
   its equilibrium geometry.    

Now, we demonstrate how to use PennyLane functionalities to implement the variational quantum
algorithm outlined above to optimize molecular geometries.

Let's get started! ⚛️

Building the parametrized electronic Hamiltonian :math:`H(x)`
-------------------------------------------------------------

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import numpy as np
from functools import partial

##############################################################################
# The second step is to specify the molecule for which we want to find the equilibrium 
# geometry. In this example, we want to optimize the geometry of the trihydrogen cation
# (:math:`\mathrm{H}_3^+`) consisted of three hydrogen atoms as shown in the figure above.
# This is done by providing a list with the symbols of the atomic species and a 1D array
# with the initial set of nuclear coordinates.

symbols = ["H", "H", "H"]
x = np.array([0.028, 0.054, 0.0,
              0.986, 1.610, 0.0,
              1.855, 0.002, 0.0])

##############################################################################
# Note that the size of the array ``x`` with the nuclear coordinates is ``3*len(symbols)``. 
# The :func:`~.pennylane_qchem.qchem.read_structure` function can also be used to read
# the molecular structure from a external file using the `XYZ format
# <https://en.wikipedia.org/wiki/XYZ_file_format>`_XYZ format or any other format
# recognized by Open Babel. For more details see the tutorial :doc:`tutorial_quantum_chemistry`.

##############################################################################
# Next, we need to build the parametrized Hamiltonian :math:`H(x)`. For a molecule, this
# is the second-quantized electronic Hamiltonian for a given set of the nuclear coordinates
# :math:`x`:
#
# .. math::
#
#     H(x) = \sum_{pq} h_{pq}(x)c_p^\dagger c_q + 
#     \frac{1}{2}\sum_{pqrs} h_{pqrs}(x) c_p^\dagger c_q^\dagger c_r c_s.
#
# In the equation above the indices of the summation run over the basis of
# Hartree-Fock molecular orbitals, the operators :math:`c^\dagger` and :math:`c` are
# respectively the electron creation and annihilation operators, and :math:`h_{pq}(x)`
# and $h_{pqrs}(x)$ are the one- and two-electron integrals carrying the dependence on
# the nuclear coordinates [#yamaguchi_book]_. The Jordan-Wigner transformation [#seeley2012]_
# is typically used to decompose the fermionic Hamiltonian into a linear combination of Pauli 
# operators,
#
# .. math::
#
#     H(x) = \sum_j h_j(x) \prod_i^{N} \sigma_i^j,
#
# whose expectation value can be evaluated using a quantum computer. $h_j(x)$ are the
# expansion coefficients inheriting the dependence on the coordinates $x$, 
# the operators $\sigma_i$ represents the Pauli group $\{I, X, Y, Z\}$ and $N$ is the
# number of qubits.
#
# We define the function ``H(x)`` to construct the parametrized qubit Hamiltonian
# of the trihydrogen cation, described in a minimal basis set, using the
# func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function.

def H(x):
    return qml.qchem.molecular_hamiltonian(symbols, x, charge=1, mapping="jordan_wigner")[0]

##############################################################################
# The variational quantum circuit
# -------------------------------
# Now, we need to define the quantum circuit to prepare the electronic ground-state
# :math:`\vert \Psi(\theta)\rangle` of the :math:`\mathrm{H}_3^+` molecule. Representing
# the wave function of this molecule requires six qubits to encode the occupation number
# of the molecular spin-orbitals that can be populated by the two electrons in the
# molecule. In order to capture the effects of electronic correlations, the :math:`N`-qubit
# system should be prepared in a superposition of the Hartree-Fock state
# :math:`\vert 110000 \rangle` with other doubly- and singly-excited configurations
# e.g., :math:`\vert 000011 \rangle`, :math:`\vert 011000 \rangle`, etc. This can be
# done using the particle-conserving single- and double-excitation gates [[#qchemcircuits]_] 
# implemented in the form of Givens rotation in PennyLane. More details on how to
# use the excitation gates to build quantum circuits for quantum chemistry applications
# see the tutorial doc:`tutorial_excitation_gates`.
#
# 
#  
#
#
# References
# ----------
#
# .. [#peruzzo2014]
#
#     A. Peruzzo, J. McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#mcardle2020]
#
#     S. McArdle, S. Endo, A. Aspuru-Guzik, S.C. Benjamin, X. Yuan, "Quantum computational 
#     chemistry". `Rev. Mod. Phys. 92, 015003  (2020).
#     <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.92.015003>`__
#
# .. [#cao2019]
#
#     Y. Cao, J. Romero, *et al.*, "Quantum chemistry in the age of quantum computing".
#     `Chem. Rev. 2019, 119, 19, 10856-10915.
#     <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__
#
# .. [#kandala2017]
#
#     A. Kandala, A. Mezzacapo *et al.*, "Hardware-efficient variational quantum
#     eigensolver for small molecules and quantum magnets". `arXiv:1704.05018
#     <https://arxiv.org/abs/1704.05018>`_
#
# .. [#romero2017]
#
#     J. Romero, R. Babbush, *et al.*, "Strategies for quantum computing molecular
#     energies using the unitary coupled cluster ansatz". `arXiv:1701.02691
#     <https://arxiv.org/abs/1701.02691>`_
#
# .. [#jensenbook]
#
#     F. Jensen. "Introduction to computational chemistry".
#     (John Wiley & Sons, 2016).
#
# .. [#fetterbook]
#
#     A. Fetter, J. D. Walecka, "Quantum theory of many-particle systems".
#     Courier Corporation, 2012.
#
# .. [#suzuki1985]
#
#     M. Suzuki. "Decomposition formulas of exponential operators and Lie exponentials
#     with some applications to quantum mechanics and statistical physics".
#     `Journal of Mathematical Physics 26, 601 (1985).
#     <https://aip.scitation.org/doi/abs/10.1063/1.526596>`_
#
# .. [#barkoutsos2018]
#
#     P. Kl. Barkoutsos, J. F. Gonthier, *et al.*, "Quantum algorithms for electronic structure
#     calculations: particle/hole Hamiltonian and optimized wavefunction expansions".
#     `arXiv:1805.04340. <https://arxiv.org/abs/1805.04340>`_
