r"""
Optimization of molecular geometries
====================================

.. meta::
    :property="og:description": Find the equilibrium geometry of a molecule
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_spectra_h2.png

.. related::
   tutorial_quantum_chemistry Quantum Chemistry with PennyLane
   tutorial_vqe Variational Quantum Eigensolver
   tutorial_givens_rotations Givens rotations for quantum chemistry
   
*Author: PennyLane dev team. Last updated: 20 May 2021.*

Predicting the most stable arrangement of the atoms in a molecule is one of the most
important tasks in computational chemistry. Typically, this is the first calculation one
has to do in order to simulate the opto-electronic and vibrational properties of molecules.
Essentially, this is an optimization problem where the total energy of the molecule is
minimized with respect to the positions of the atomic nuclei.

In the framework of the `Born-Oppenheimer approximation <https://en.wikipedia.org/wiki/
Born%E2%80%93Oppenheimer_approximation>`_ [#kohanoff2006]_ the total electronic energy of the 
molecule :math:`E(x)` depends on the nuclear coordinates :math:`x` which defines
the potential energy surface (PES). Solving the stationary problem :math:`\nabla_x E(x) = 0` 
corresponds to what is known as *molecular geometry optimization* and the optimized nuclear
coordinates determine the *equilibrium geometry* of the molecule. For example, the figure below
illustrates these concepts for the
`trihydrogen cation <https://en.wikipedia.org/wiki/Trihydrogen_cation>`_. Its equilibrium
geometry in the electronic ground-state corresponds to the energy minimum in the PES
where the three hydrogen atoms are located at the vertices of an equilateral
triangle whose side length is the optimized H-H bond length :math:`d`.

|

.. figure:: /demonstrations/mol_geo_opt/fig_pes.png
    :width: 50%
    :align: center

|

Classical algorithms for molecular geometry optimization are computationally expensive. Typically,
they rely on the Newton-Raphson method [#jensenbook]_ which requires access to the
gradient and the Hessian of the energy with respect to the nuclear coordinates at each
optimization step. As a consequence, using accurate wave function methods to solve the
molecule's electronic structure while the global minimum is searched along the
multidimentional PES is computationally intractable even for medium-size molecules.
In practice, `density functional theory <https://en.wikipedia.org/wiki/ensity_functional_theory>`_ methods [#dft_book]_ are used to obtain approximated geometries.


On the other hand, variational quantum algorithms for quantum chemistry simulations use a
quantum computer to prepare the electronic wave function of a molecule and to measure the
expectation value of the Hamiltonian while a classical optimizer adjusts the circuit parameters
to minimize the total energy [#mcardle2020]_. In this tutorial we learn how to recast the
problem of finding the equilibrium geometry of a molecule in terms of a more general
variational quantum algorithm. In this case, we consider explicitly that the target
electronic Hamiltonian :math:`H(x)` is a *parametrized* observable that depends on
the nuclear coordinates :math:`x`. This implies that the objective function, defined
by the expectation value of the Hamiltonian :math:`H(x)` computed in the trial state
:math:`\vert \Psi(\theta) \rangle` prepared by a quantum circuit, depends on both the
circuit and the Hamiltonian parameters. In addition, we minimize the generalized cost
function using a *joint* optimization scheme where the gradients of the cost function
with respect to circuit parameters and nuclear coordinates are simultaneously computed
at each optimization step. Interestingly, this approach does not require nested electronic
structure calculations for each set of nuclear coordinates, as occurs in the analogous
classical algorithms. Once the optimization is finalized, the circuit parameters
determine the energy of the electronic state, and the nuclear coordinates the
equilibrium geometry of the molecule in this electronic state.

Here we demonstrate how to use PennyLane functionalities to implement a
variational quantum algorithm to optimize the geometry of a molecule.
The quantum algorithm will be described as follows:

#. Define the molecule for which we want to find the equilibrium geometry.

#. Build the parametrized electronic Hamiltonian :math:`H(x)` for a given set
   of nuclear coordinates :math:`x`.

#. Design the variational quantum circuit to prepare the electronic state of the
   molecule :math:`\vert \Psi(\theta) \rangle`.

#. Define the cost function :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert
   \Psi(\theta) \rangle`.

#. Initialize the variational parameters :math:`\theta` and :math:`x` and minimize
   the cost function :math:`g(\theta, x)`

Let's get started! ⚛️

Building the parametrized electronic Hamiltonian
------------------------------------------------

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import numpy as np
from functools import partial
import matplotlib.pyplot as plt

##############################################################################
# Now, we specify the molecule for which we want to find the equilibrium
# geometry. In this example, we want to optimize the geometry of the trihydrogen cation
# (:math:`\mathrm{H}_3^+`) consisted of three hydrogen atoms (see figure above).
# This is done by providing a list with the symbols of the atomic species and a
# 1D array with the initial set of nuclear coordinates in
# `atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_ (Bohr radii).

symbols = ["H", "H", "H"]
x = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])

##############################################################################
# The size of the array ``x`` with the nuclear coordinates is ``3*len(symbols)``.
# The :func:`~.pennylane_qchem.qchem.read_structure` function can also be used to read
# the molecular structure from a external file using the `XYZ format
# <https://en.wikipedia.org/wiki/XYZ_file_format>`_ XYZ format or any other format
# recognized by Open Babel.
# For more details see the tutorial :doc:`tutorial_quantum_chemistry`.

##############################################################################
# Next, we need to build the parametrized Hamiltonian :math:`H(x)`. For a molecule,
# this is the second-quantized electronic Hamiltonian for a given set of the
# nuclear coordinates :math:`x`:
#
# .. math::
#
#     H(x) = \sum_{pq} h_{pq}(x)c_p^\dagger c_q +
#     \frac{1}{2}\sum_{pqrs} h_{pqrs}(x) c_p^\dagger c_q^\dagger c_r c_s.
#
# In the equation above the indices :math:`p, q, r, s` run over the basis of
# Hartree-Fock molecular orbitals, the operators :math:`c^\dagger` and :math:`c` are
# respectively the electron creation and annihilation operators, and :math:`h_{pq}(x)`
# and :math:`h_{pqrs}(x)` are the one- and two-electron integrals carrying the dependence on
# the nuclear coordinates [#yamaguchi_book]_. The Jordan-Wigner transformation [#seeley2012]_
# is typically used to decompose the fermionic Hamiltonian into a linear combination of Pauli
# operators,
#
# .. math::
#
#     H(x) = \sum_j h_j(x) \prod_i^{N} \sigma_i^j,
#
# whose expectation value can be evaluated using a quantum computer. The expansion
# coefficients :math:`h_j(x)` inherit the dependence on the coordinates :math:`x`,
# the operators :math:`\sigma_i` represents the Pauli group :math:`\{I, X, Y, Z\}` and
# :math:`N` is the number of qubits required to represent the electronic wave function
# as we will discuss in the next section.
#
# We define the function ``H(x)`` to build the parametrized qubit Hamiltonian
# of the trihydrogen cation, described with a minimal basis set, using the
# :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function.

def H(x):
    return qml.qchem.molecular_hamiltonian(symbols, x, charge=1, mapping="jordan_wigner")[0]

##############################################################################
# Note that we have used the keyword argument ``charge`` to specify the net charge
# of the molecule. The :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function
# allows us the user to define other keyword arguments to generate the Hamiltonian
# of more complicated systems.
#
# The variational quantum circuit
# -------------------------------
#
# Now, we need to define the quantum circuit to prepare the electronic ground-state
# :math:`\vert \Psi(\theta)\rangle` of the :math:`\mathrm{H}_3^+` molecule. Representing
# the wave function of this molecule requires six qubits to encode the occupation number
# of the molecular spin-orbitals which can be populated by the two electrons in the
# molecule. To capture the effects of electronic correlations [#kohanoff2006]_,
# we need to prepare the :math:`N`-qubit in a superposition of the Hartree-Fock (HF) state
# :math:`\vert 110000 \rangle` with other states that differ by a double- or
# single-excitation with respect to the HF state. For example, the state
# :math:`\vert 000011 \rangle` is obtained by exciting two particles from qubits 0, 1 to
# 4, 5. Similarly, the state :math:`\vert 011000 \rangle` corresponds to a single excitation
# from qubit 0 to 2. This can be done using the single-excitation and
# double-excitation gates :math:`G` and :math:`G^{(2)}` [#qchemcircuits]_ implemented
# in the form of Givens rotations in PennyLane. For more details see the tutorial
# :doc:`tutorial_givens_rotations`.
#
# In addition, here we use an adaptive algorithm to select the excitation
# operations included in the variational quantum circuit. The algorithm, which is
# described in more details in the tutorial :doc:`tutorial_adaptive_algorithm`,
# proceeds as follows:
#
# #. Generate the indices of the qubits involved in all single- and
#    double-excitations using the :func:`~.pennylane_qchem.qchem.excitation` function.
#    For example, the indices of the singly-excited state :math:`\vert 011000 \rangle`
#    are given by the list ``[0, 2]``. Similarly, the indices of the doubly-excited
#    state :math:`\vert 000011 \rangle` are ``[0, 1, 4, 5]``.
#
# #. Construct the circuit using all double-excitation gates. Compute the gradient
#    of the cost function :math:`g(\theta, x)` with respect to each double-excitation
#    gate and retain only those with non-zero gradient.
#
# #. Include the selected double-excitation gates and repeat the process for the
#    single-excitation gates.
#
# #. Build the final variational quantum circuit by including the selected gates.
#
# For the :math:`\mathrm{H}_3^+` molecule in a minimal basis set we have a total of eight
# excitations of the HF reference state. After applying the adaptive algorithm the final
# quantum circuit contains only two double-excitation operations that act on the qubits
# ``[0, 1, 2, 3]`` and ``[0, 1, 4, 5]``. The circuit is shown in the figure below.
#
# |
#
# .. figure:: /demonstrations/mol_geo_opt/fig_circuit.png
#     :width: 70%
#     :align: center
#
# |
#
# This quantum circuit is implemented by the ``circuit`` function below

hf = qml.qchem.hf_state(electrons=2, orbitals=6)

def circuit(params, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])

##############################################################################
# The ``DoubleExcitation`` operations acting on the HF state prepare
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
# optimized to find the ground-state equilibrium geometry of the molecule.
#
# The cost function and the nuclear gradients
# -------------------------------------------
#
# The next step is to define the cost function
# :math:`g(\theta, x) = \langle \Psi(\theta) \vert H(x) \vert\Psi(\theta) \rangle`. It
# evaluates the expectation value of the *parametrized* Hamiltonian :math:`H(x)` in the
# trial state :math:`\vert\Psi(\theta)\rangle`. First, we define the quantum device used
# to compute the expectation value. In this example, we use the ``default.qubit``
# simulator of PennyLane:

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
# In order to minimize the cost function :math:`g(\theta, x)` we use a gradient-based
# method. To that aim, we have to compute the gradients with respect to the both the
# circuit parameters :math:`\theta` *and* the nuclear coordinates :math:`x` (nuclear gradients).
# The circuit gradients are computed analytically using the automatic differentiation
# techniques available in PennyLane. On the other hand, the nuclear gradients are evaluated
# by taking the expectation value of the gradient of the electronic Hamiltonian,
#
# .. math::
#
#     \nabla_x g(\theta, x) = \langle \Psi(\theta) \vert \nabla_x H(x) \vert \Psi(\theta) \rangle.
#
# We use the :func:`~.pennylane.finite_diff` function to compute the gradient of
# the Hamiltonian using a central-difference approximation. Then, the PennyLane class
# :class:`~.pennylane.ExpvalCost` is used to evaluate the expectation value of
# the gradient components :math:`\frac{\partial H(x)}{\partial x_i}`. This is implemented by
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
# equilibrium geometry of the :math:`\mathrm{H}_3^+` molecule. As a reminder, 
# the circuit parameters and the nuclear coordinates will be jointly optimized at
# each optimization step. Note that this approach does not require a nested VQE
# optimizations of the circuit parameters for each set of the nuclear coordinates.
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
# the initial electronic state :math:`\vert\Psi(\theta_1, \theta_2)\rangle` 
# is approximated to the Hartree-Fock (HF) state. The initial set of nuclear
# coordinates :math:`x`, defined at the beginning of the tutorial, was computed
# classically within the HF approximation using the GAMESS program [#ref_gamess]_.
# This is a natural choice for the starting geometry that we are aiming to correct
# due to the electronic correlation effects included in the trial state
# :math:`\vert\Psi(\theta)\rangle`.
#
# We carry out the optimization over a maximum of 100 steps. In this example
# The circuit parameters and the nuclear coordinates are optimized until the
# maximum component of the nuclear gradient :math:`\nabla_x g(\theta,x)` is
# less than or equal to :math:`10^{-5}` Hartree/Bohr. Typically, this is the
# convergence criterion used for optimizing molecular geometries in
# quantum chemistry simulations.
#
# Finally, we use the lists ``energy`` and ``bond_length`` to keep track of the
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

    if n % 2 == 0:
        print(
            "Iteration = {:},  Energy = {:.8f} Ha,  bond length = {:.5f} A".format(
                n, energy[-1], bond_length[-1]
            )
        )

    if np.max(grad_fn(x)) <= 1e-05:
        break

print("\n" "Final value of the ground-state energy = {:.8f} Ha".format(energy[-1]))
print("\n" "Ground-state equilibrium geometry")
print("%s %4s %8s %8s" % ("symbol", "x", "y", "z"))
for i, atom in enumerate(symbols):
    print("  {:}    {:.4f}   {:.4f}   {:.4f}".format(atom, x[3 * i], x[3 * i + 1], x[3 * i + 2]))

##############################################################################
# Next, we plot the values of the ground state energy of the molecule, relative
# to the exact value computed with the full configuration interaction (FCI) method,
# and the of H-H bond lengths as the circuit parameters and the nuclear coordinates are
# optimized by the variational quantum algorithm.

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Adds energy plot on column 1
E_fci = -1.27443765658
E_vqe = np.array(energy)
ax1 = fig.add_subplot(121)
ax1.plot(range(n+1), E_vqe-E_fci, 'go-', ls='dashed')
ax1.plot(range(n+1), np.full(n+1, 0.001), color='red')
ax1.set_xlabel("Optimization step", fontsize=13)
ax1.set_ylabel("$E_{VQE} - E_{FCI}$ (Hartree)", fontsize=13)
ax1.text(5, 0.0013, r'Chemical accuracy', fontsize=13)
plt.yscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adds bond length plot on column 2
d_fci = 0.986
ax2 = fig.add_subplot(122)
ax2.plot(range(n+1), bond_length, 'go-', ls='dashed')
ax2.plot(range(n+1), np.full(n+1, d_fci), color='red')
ax2.set_ylim([0.968,0.99])
ax2.set_xlabel("Optimization step", fontsize=13)
ax2.set_ylabel("H-H bond length ($\AA$)", fontsize=13)
ax2.text(5, 0.9865, r'Equilibrium bond length', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3)
plt.show()

##############################################################################
# |
# Notice that despite the fact that the ground-state energy is already converged
# within chemical accuracy (:math:`0.001` Ha) after the fourth iteration, more
# optimization steps are required to find the equilibrium bond length of the
# molecule.
#
# The figure below animates snapshots of the atomic structure of the
# trihydrogen cation as the quantum algorithm was searching for the equilibrium
# geometry. For visualization purposes, the initial nuclear coordinates were
# generated by perturbing the HF geometry. Note that the quantum algorithm
# is able to find the correct optimized geometry of the :math:`\mathrm{H}_3^+`
# molecule where the three H atoms are located at the vertices of an equilateral triangle.
#
# |
#
# .. figure:: /demonstrations/mol_geo_opt/fig_movie.gif
#     :width: 50%
#     :align: center
#
# |
#
# To summarize, this tutorial shows that the scope of variational quantum algorithms can be
# extended to perform quantum simulations of molecules involving both the electronic and
# the nuclear degrees of freedom. The joint optimization scheme described here
# is a generalization of the usual VQE algorithm where only the electronic
# state is parametrized. By using the adaptive algorithm it is possible to reduce the
# number of excitation gates in the quantum circuit to apply this algorithm to molecules
# of increasing complexity [#geo_opt_paper]_.
#
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
#
# .. [#geo_opt_paper]
#
#     A. Delgado, J.M. Arrazola, S. Jahangiri, Z. Niu, J. Izaac, C. Roberts, N. Killoran.
#     "Variational quantum algorithm for molecular geometry optimization".
#     arXiv preprint
