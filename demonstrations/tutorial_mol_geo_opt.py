r"""
Optimization of molecular geometries
====================================

.. meta::
    :property="og:description": Find the equilibrium geometry of a molecule
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_spectra_h2.png

.. related::
   tutorial_vqe Variational Quantum Eigensolver
   
*Author: PennyLane dev team. Last updated: 17 May 2021.*

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
intractable even for medium-size molecules. Instead, `density functional theory
 <https://en.wikipedia.org/wiki/Density_functional_theory>`_ methods [#dft_book]_ are
used to obtain approximated geometries.

Variational quantum algorithms for quantum chemistry applications use quantum computer to prepare
the electronic wave function of a molecule and to measure the expectation value of the Hamiltonian
while a classical optimizer adjusts the circuit parameters in order to minimize the total
energy [#mcardle2020]_. The problem of finding the equilibrium geometry of a molecule can be 
recast in terms of a more general variational quantum algorithm where the target electronic 
Hamiltonian :math:`H(x)` is a *parametrized* observable that depends on the nuclear
coordinates :math:`x`. This implies that the objective function, defined by the expectation value
of the Hamiltonian :math:`H(x)` computed in the trial state :math:`\vert \Psi(\theta) \rangle`
prepared by a parametrized quantum circuit :math:`U(\theta)`, depends on both the circuit and the 
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
x = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])

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
#
# Now, we need to define the quantum circuit to prepare the electronic ground-state
# :math:`\vert \Psi(\theta)\rangle` of the :math:`\mathrm{H}_3^+` molecule. Representing
# the wave function of this molecule requires six qubits to encode the occupation number
# of the active spin-orbitals which can be populated by the two electrons in the
# molecule. In order to capture the effects of electronic correlations [#kohanoff2006]_,
# we need to prepare the :math:`N`-qubit in a superposition of the Hartree-Fock state
# :math:`\vert 110000 \rangle` with other states that differ by a double- or
# single-excitation with respect to the HF state. For example, the state
# :math:`\vert 000011 \rangle` is obtained by exciting two particles from qubits  0, 1 to
# 4, 5. Similarly, the state encodes a double excitation of the reference HF state
# where the state :math:`\vert 011000 \rangle` corresponds to a single excitation
# from qubit 0 to 2. This can be done using the particle-conserving single- and
# double-excitation gates [#qchemcircuits]_ implemented in the form of Givens rotations
# in PennyLane. For more details see the tutorial doc:`tutorial_givens_rotations`.
#
# Here, we use an adaptive algorithm to select the excitation
# operation included in the variational quantum circuit. The algorithm, which is
# described in more details in the tutorial doc:`tutorial_adaptive_algorithm`,
# proceeds as follows:
#
# #. Generate the lists with the indices of the qubits involved in all single- and
#    double-excitations using the func:`~.pennylane_qchem.qchem.excitation` function.
#    For example, the indices of the singly-excited state :math:`\vert 011000 \rangle`
#    are given by the list ``[0, 2]``. Similarly, the indices of the doubly-excited
#    state :math:`\vert 000011 \rangle` are ``[0, 1, 4, 5]``.
#
# #. Construct the circuit using all double-excitation gates. Compute the gradient
#    of the expectation value
#    :math:`\langle \Psi(\theta) \vert H(x) \vert \Psi(\theta) \rangle` with respect
#    to each double-excitation gate and retain only those with non-zero gradient.
#
# #. Include the selected double-excitation gates and repeat the process for the
#    single-excitation gates.
#
# #. Build the final variational quantum circuit by including the selected excitation
#    operations.
#
# For the :math:`\mathrm{H}_3^+` molecule in a minimal basis set we have a total of eight
# excitation operations. After applying the adaptive algorithm the final quantum
# circuit contains two double-excitation operations that act on the qubits ``[0, 1, 2, 3]``
# and ``[0, 1, 4, 5]``. The circuit is shown in the figure below.
#
# |
#
# .. figure:: /demonstrations/mol_geo_opt/fig_circuit.png
#     :width: 50%
#     :align: center
#
# |
#
# The quantum circuit above is implemented by the ``circuit`` function


def ansatz(params, wires):
    hf_state = np.array([1, 1, 0, 0, 0, 0])
    qml.BasisState(hf_state, wires=wires)
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])


##############################################################################
# The ``DoubleExcitation`` operations acting on the HF state allow us to prepare
# the trial state
#
# .. math::
#
#     \vert\Psi(\theta_1, \theta_2)\rangle =
#     \mathrm{cos}(\theta_1)\mathrm{cos}(\theta_2)\vert110000\rangle -
#     \mathrm{cos}(\theta_1)\mathrm{sin}(\theta_2)\vert000011\rangle -
#     \mathrm{sin}(\theta_1)\vert001100\rangle,
#
# where :math:`\theta_1` and :math:`\theta_2` are the circuit parameters that need to be
# optimized to find the electronic ground state of the trihydrogen cation.
#
##############################################################################
# The cost function and the nuclear gradients
# -------------------------------------------
#
# The next step is to define the cost function
# :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert\Psi(\theta) \rangle` to
# evaluate the expectation value of the *parametrized* Hamiltonian :math:`H(x)` in the
# trial state :math:`\vert\Psi(\theta)\rangle`. First, we define the device to compute
# the expectation value. In this example, we use PennyLane's qubit simulator:

dev = qml.device("default.qubit", wires=6)

##############################################################################
# Next, we use the PennyLane class :class:`~.pennylane.ExpvalCost` to define the
# ``cost`` function :math:`g(\theta, x)` which depends on both the circuit and the
# Hamiltonian parameters.


def cost(params, x):
    return qml.ExpvalCost(circuit, H(x), dev)(params)


##############################################################################
# This function returns the expectation value of the parametrized Hamiltonian ``H(x)``
# computed in the state prepared by the ``circuit`` function for a given set of the
# circuit parameters ``params`` and the nuclear coordinates ``x``.
#
# In order to to minimize the cost function :math:`g(\theta, x)` using a gradient-based
# method we have to compute the gradients with respect to the both the circuit parameters
# :math:`\theta` *and* the nuclear coordinates :math:`x` (nuclear gradients). The circuit
# gradients are computed analytically using the automatic differentiation techniques available
# in PennyLane. On the other hand, the nuclear gradients are evaluated by taking the
# expectation value of the gradient of the electronic Hamiltonian,
#
# .. math::
#
#     \nabla_x g(\theta, x) = \langle \Psi(\theta) \vert \nabla_x H(x) \vert \Psi(\theta) \rangle.
#
# We use the func:`~.pennylane.finite_diff` function to compute the gradient of
# the Hamiltonian using a central-difference approximation and the PennyLane class
# :class:`~.pennylane.ExpvalCost` to evaluate the expectation value of
# the gradient components :math:`\frac{\partial H(x)}{\partial x_i}. This is implemented by
# the function ``grad_x``:


def grad_x(x, params):
    grad_h = qml.finite_diff(H)(x)
    grad = [qml.ExpvalCost(circuit, obs, dev)(params) for obs in grad_h]
    return np.array(grad)


##############################################################################
# Optimization of the molecular geometry
# --------------------------------------
#
# Now we proceed to minimize our cost function to find the ground state energy and the
# equilibrium geometry of the :math:`\mathrm{H}_3^+` molecule. The circuit parameters and
# the nuclear coordinates will be jointly optimized at each optimization step. We note
# that this approach does not require a nested VQE optimization of the circuit parameters
# for each set of the nuclear coordinates.
#
# We start by defining the classical optimizers:

opt_theta = qml.GradientDescentOptimizer(stepsize=0.4)
opt_x = qml.GradientDescentOptimizer(stepsize=0.8)

##############################################################################
# In this example, we are using a simple gradient-descent optimizer to update
# the circuit and the Hamiltonian parameters during the iterative optimization.
# Next, we initialize the circuit parameters :math:`\theta`

theta = [0.0, 0.0]

##############################################################################
# Setting the angles :math:`\theta_1` and :math:`\theta_2` to zero implies that
# the initial electronic state :math:`\vert\Psi(\theta_1, \theta_2)\rangle`prepared
# by the ``circuit`` function is the Hartree-Fock (HF) state. The initial set of nuclear
# coordinates :math:`x`, defined at the beginning of the tutorial, was computed
# classically within the HF approximation using the GAMESS program [#ref_gamess]_.
#
# We carry out the optimization over a maximum of 100 steps. The circuit parameters and
# the nuclear coordinates should be optimized until the maximum component of the
# nuclear gradient :math:`\nabla_x g(\theta,x)` is less than or equal to
# :math:`10^{-5}` Hartree/Bohr. Typically, this is the convergence criterion used for
# optimizing molecular geometries in quantum chemistry simulations.
#
# Finally, we use the lists ``energy`` and ``bond_length`` to keep track the
# value of the cost function :math:`g(\theta,x)` and the H-H bond length :math:`d`
# (in Angstroms) of trihydrogen cation through the iterative procedure.

energy = []
bond_length = []
bohr_angs = 0.529177210903

for n in range(100):

    theta = opt_theta.step(partial(cost, x=x), theta)

    grad_fn = partial(grad_x, params=theta)
    x = opt_x.step(partial(cost, params=theta), x, grad_fn=grad_fn)

    energy.append(cost(theta, x))
    bond_length.append(np.linalg.norm(x[0:3] - x[3:6]) * bohr_angs)

    print(
        "Iteration = {:},  Energy = {:.8f} Ha,  bond length = {:.4f} A".format(
            n, energy[-1], bond_length[-1]
        )
    )

    if np.max(grad_fn(x)) <= 1e-04:
        break

print("\n" "Final value of the ground-state energy = {:.8f} Ha".format(energy[-1]))
print("\n" "Ground-state equilibrium geometry")
print("%s %4s %8s %8s" % ("symbol", "x", "y", "z"))
for i, atom in enumerate(symbols):
    print("  {:}    {:.4f}   {:.4f}   {:.4f}".format(atom, x[3 * i], x[3 * i + 1], x[3 * i + 2]))

##############################################################################
# References
# ----------
#
# .. [#kohanoff2006]
#
#     Jorge Kohanoff. "Electronic structure calculations for solids and molecules: theory and
#     computational methods". (Cambridge University Press, 2006).
#
# .. [#jensenbook]
#
#     F. Jensen. "Introduction to computational chemistry".
#     (John Wiley & Sons, 2016).
#
# .. [#dft_book]
#
#     W. Koch, M.C. Holthausen. "A Chemist's Guide to Density Functional Theory".
#     (John Wiley & Sons, 2015).
#
# .. [#mcardle2020]
#
#     S. McArdle, S. Endo, A. Aspuru-Guzik, S.C. Benjamin, X. Yuan, "Quantum computational
#     chemistry". `Rev. Mod. Phys. 92, 015003  (2020).
#     <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.92.015003>`__
#
# .. [#yamaguchi_book]
#
#     Y. Yamaguchi, H.F. Schaefer. "A New Dimension to Quantum Chemistry: Analytic Derivative
#     Methods in *Ab Initio* Molecular Electronic Structure Theory".
#     (Oxford University Press, USA, 1994).
#
# .. [#seeley2012]
#
#     Jacob T. Seeley, Martin J. Richard, Peter J. Love. "The Bravyi-Kitaev transformation for
#     quantum computation of electronic structure". `Journal of Chemical Physics 137, 224109
#     (2012).
#     <https://aip.scitation.org/doi/abs/10.1063/1.4768229>`__
#
# .. [#qchemcircuits]
#
#     J.M. Arrazola, O. Di Matteo, N. Quesada, S. Jahangiri, A. Delgado, N. Killoran.
#     "Universal quantum circuits for quantum chemistry".
#     arXiv preprint
#
# .. [#ref_gamess]
#
#     M.W. Schmidt, K.K. Baldridge, J.A. Boatz, S.T. Elbert, M.S. Gordon, J.H. Jensen,
#     S. Koseki, N. Matsunaga, K.A. Nguyen, S.Su, *et al.* "General atomic and molecular
#     electronic structure system". `Journal of Computational Chemistry 14, 1347 (1993)
#     <https://onlinelibrary.wiley.com/doi/10.1002/jcc.540141112>`__
