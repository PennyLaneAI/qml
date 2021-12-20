r"""

Autodifferentiable Hartree-Fock calculations
============================================

.. meta::
    :property="og:description": Learn how to use the differentiable Hartree-Fock solver
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry


*Author: PennyLane dev team. Posted:  2021. Last updated: XX December 2021*

Variational quantum algorithms aim to calculate the energy of a molecule by constructing a
parametrized quantum circuit and finding a set of parameters that minimize the expectation value of
the electronic molecular Hamiltonian. For a given molecule, the electronic Hamiltonian is obtained
by solving the Hartree-Fock equations which provide a set of integrals that are used to construct
the Hamiltonian. The constructed Hamiltonian depends on molecular parameters such as atomic
coordinates and basis set parameters that can also be optimized concurrently with the circuit
parameters. This can be beneficial to solve problems such as molecular geometry optimization where
the atomic coordinates and the circuit parameters can be optimized simultaneously to obtain the
equilibrium geometry of a molecule at the ground or excited electronic state. The optimization
problem can now be solved by differentiating the expectation value of the Hamiltonian with respect
to the circuit and molecular parameters and using an algorithm such as
`gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ to obtain the set of parameters
that minimize the following expectation value

   .. math::

       \left \langle \Psi(\theta) | H(\phi) | \Psi(\theta) \right \rangle,

where :math:`\theta` and :math:`\phi` represent the circuit and molecular parameters, respectively.
The ability to optimize the Hamiltonian parameters provides computational advantages for geometry
optimization and also allows customizing basis set parameters for a given molecule such that a lower
energy cn be reached with the same number of basis functions.

The gradients of the Hamiltonian expectation value can be obtained from symbolic, numeric, or
automatic differentiation. Symbolic differentiation obtains derivatives of an input function by
direct mathematical manipulation, for example using standard strategies of differential calculus.
These can be performed by hand or with the help of computer algebra software. The resulting
expressions are exact, but the symbolic approach is of limited scope, particularly since many
functions are not known in explicit analytical form. Symbolic methods also suffer from the
expression swell problem where careless usage can lead to exponentially large symbolic expressions.
Numerical differentiation is a versatile but unstable method, often relying on finite differences to
calculate approximate derivatives. This is problematic especially for large computations consisting
of many differentiable parameters. Automatic differentiation is a computational strategy where a
function implemented using computer code is differentiated by expressing it in terms of elementary
operations for which derivatives are known. The gradient of the target function is then obtained by
applying the chain rule through the entire code. In principle, automatic differentiation can be used
to calculate derivatives of a function using resources comparable to those required to evaluate the
function itself.

In this tutorial, you will learn how to use the autodifferentiable Hartree-Fock solver implemented
in PennyLane. The Hartree-Fock module in PennyLane provides built-in methods for constructing atomic
and molecular orbitals, building Fock matrices, and solving the self-consistent Hartree-Fock
equations to obtain optimized orbitals, which can be used to construct fully-differentiable
molecular Hamiltonians. PennyLane allows users to natively compute derivatives of all these objects
with respect to the underlying parameters using the methods of automatic differentiation. We also
introduce a workflow to jointly optimize circuit parameters, nuclear coordinates, and basis set
parameters in a variational quantum eigensolver algorithm. Let's get started!

The Hartree-Fock method
-------------------

The main goal of the Hartree-Fock method is to obtain molecular spin-orbitals that minimize the
energy of a system where electrons are treated as independent particles that experience the effect
of the other electrons in an average way. These optimized molecular orbitals are then used to
construct one- and two-body electron integrals in the molecular orbital basis which can be used to
generate differentiable second-quantized molecular Hamiltonians in the fermionic and qubit basis.
In this section we show how to use the PennyLane's differentiable Hartree-Fock solver.

To get started, we need to define the atomic symbols and coordinates of the desired molecule. For
the hydrogen molecule we define the symbols and geometry as
"""

import autograd
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

symbols = ["H", "H"]
geometry = np.array([[-0.672943567415407, 0.0, 0.0],
                     [ 0.672943567415407, 0.0, 0.0]], requires_grad=True)

##############################################################################
# Note that `requires_grad=True` specifies that the atomic coordinates are differentiable
# parameters that we can compute gradients with with respect to them. We can now compute the
# Hartree-Fock energy and its gradient with respect to the atomic coordinates for hydrogen. To do
# that, we need to create a molecule object which stores all the molecular parameters needed to
# perform a Hartree-Fock calculation.

mol = qml.hf.Molecule(symbols, geometry)

##############################################################################
# The Hartree-Fock energy can now be computed with the :func:`~.pennylane.hf.hf_energy` function

qml.hf.hf_energy(mol)(geometry)

##############################################################################
# The computed energy matches the reference value of :math:`-1.1175058849` Ha. We now compute the
# gradient of the energy with respect to the atomic coordinates with Autograd

autograd.grad(qml.hf.hf_energy(mol))(geometry)

##############################################################################
# Note that we need to pass the `mol` object and the geometry at which the gradients are computed.
# The obtained gradients are equal or very close to zero because the geometry we used here has been
# previously optimized at the Hartree-Fock level. You can use a different geometry and verify that
# the newly computed  gradients are not all zero.
#
# We can also compute the values and gradients of several other quantities that are computed during
# the Hartree-Fock procedure. These include all integrals over basis functions, matrices formed from
# these integrals and the one- and two-body integrals over molecular orbitals [#arrazola2021]_.
# Let's look at few examples.
#
# We first compute the overlap integral between the two S-type atomic orbitals of the hydrogen
# atoms. Here we are using the `3to-3g <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`_
# basis set in which each of the atomic orbitals is represented by one basis function composed of
# three contracted Gaussian functions. These basis functions can be accessed from the molecule
# object as

S1 = mol.basis_set[0]
S2 = mol.basis_set[1]

##############################################################################
# We can check the parameters of the basis functions as

S1.params

##############################################################################
# This returns the exponents, contraction coefficients and the centres of the three Gaussian
# functions of the sto-3g basis set. These parameters can be also obtained individually by using
# `S1.alpha`, `S1.coeff` and `S1.r`, respectively. You can verify that both of the S1 and S2
# atomic orbitals have the same exponents and contraction coefficients but are centred on different
# hydrogen atoms. You can also verify that the orbitals are S-type by printing the angular momentum
# quantum numbers with

S1.l

##############################################################################
# This gives you a tuple of three integers, representing the exponents of the `x`, `y` and `z`
# components in the Gaussian functions [#arrazola2021]_.
#
# Having the two atomic orbitals, we can now compute the overlap integral by passing the orbitals
# and the initial values of their centres to the :func:`~.pennylane.hf.generate_overlap` function.
# Note that the centres of the orbitals are those of the hydrogen atoms by default and are therefore
# treated as differentiable parameters by PennyLane.

qml.hf.generate_overlap(S1, S2)([geometry[0], geometry[1]])

##############################################################################
# You can verify that the overlap integral between two identical atomic orbitals is one. We can now
# compute the gradient of the overlap integral with respect to the orbital centres

autograd.grad(qml.hf.generate_overlap(S1, S2))([geometry[0], geometry[1]])

##############################################################################
# Can you explain why some of the computed gradients are zero?
#
# Let's now do a cool thing and plot the atomic orbitals and their overlap. We can do it by using
# the :func:`~.pennylane.hf.molecule.atomic_orbital` function. This function computes the actual
# value of the atomic orbital at a given coordinate. For instance, the value of the S orbital on the
# first hydrogen atom can be computed at the origin as

S1 = mol.atomic_orbital(0)
S1(0.0, 0.0, 0.0)

##############################################################################
# We can compute the value of this orbital on different points along the `x` axis and plot it.

x = np.linspace(-5, 5, 1000)
plt.plot(x, S1(x, 0.0, 0.0), color='teal')
plt.xlabel('X [Bohr]')
plt.show()

##############################################################################
# We can also plot the second S orbital and visualize the overlap between them

S2 = mol.atomic_orbital(1)
plt.plot(x, S1(x, 0.0, 0.0), color='teal')
plt.plot(x, S2(x, 0.0, 0.0), color='teal')
plt.fill_between(
    x,  np.minimum(S1(x, 0.0, 0.0), S2(x, 0.0, 0.0)), color = 'red', alpha = 0.5, hatch = '||')
plt.xlabel('X [Bohr]')
plt.show()

##############################################################################
# By looking at the orbitals, can you guess at what distance the value of the overlap becomes
# negligible? Can you verify your guess by computing the overlap at that distance?
#
# Similarly, we can plot the molecular orbitals of the hydrogen molecule obtained from the
# Hartree-Fock calculations. We plot the cross section of the bonding orbital on the `x-y` plane.

n = 30 # number of grid points along each axis

mol.mo_coefficients = mol.mo_coefficients.T
mo = mol.molecular_orbital(0)
x, y = np.meshgrid(np.linspace(-2, 2, n),
                   np.linspace(-2, 2, n))
val = np.vectorize(mo)(x, y, 0)
val = np.array([val[i][j]._value for i in range(n) for j in range(n)]).reshape(n, n)

fig, ax = plt.subplots()
co = ax.contour(x, y, val, 10, cmap='summer_r', zorder=0)
ax.clabel(co, inline=2, fontsize=10)
plt.scatter(mol.coordinates[:,0], mol.coordinates[:,1], s = 80, color='black')
ax.set_xlabel('X [Bohr]')
ax.set_ylabel('Y [Bohr]')
plt.show()

##############################################################################
# VQE simulations
# ---------------
#
# By performing the Hartree-Fock calculations, we obtain a set of one- and two-body integrals
# over molecular orbitals that can be used to construct the molecular Hamiltonian [#arrazola2021]_
# with the :func:`~.pennylane.hf.generate_hamiltonian` function.

hamiltonian = qml.hf.generate_hamiltonian(mol)(geometry)
print(hamiltonian)

##############################################################################
# The Hamiltonian contains 15 terms and, importantly, the coefficients of the Hamiltonian are all
# differentiable. We can construct a circuit and perform a VQE simulation in which both of the
# circuit parameters and the atomic coordinates are optimized simultaneously by using gradients
# computed with Autograd. We will have two sets of differentiable parameters: the first set is the
# rotation angles of the `Excitation` gates which are applied to the reference Hartree-Fock state
# to construct the exact ground state and the second set contains the atomic coordinates of the
# hydrogen atoms.

dev = qml.device("default.qubit", wires=4)
def energy(mol):
    @qml.qnode(dev)
    def circuit(*args):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
        qml.DoubleExcitation(*args[0][0], wires=[0, 1, 2, 3])
        return qml.expval(qml.hf.generate_hamiltonian(mol)(*args[1:]))
    return circuit

##############################################################################
# Note that we only use the `DoubleExcitation` gate as the `SingleExcitation` ones can be neglected
# in this particular example [#szabo1996]_. We now compute the gradients of the energy with respect
# to the circuit parameter and the atomic coordinates and update the parameters iteratively. Note
# that the atomic coordinate gradients are simply the forces on the atomic nuclei.

# initial value of the circuit parameter
circuit_param = [np.array([0.0], requires_grad=True)]

for n in range(36):

    args = [circuit_param, geometry]
    mol = qml.hf.Molecule(symbols, geometry)

    # gradient for circuit parameters
    g_param = autograd.grad(energy(mol), argnum = 0)(*args)
    circuit_param = circuit_param - 0.25 * g_param[0]

    # gradient for nuclear coordinates
    forces = -autograd.grad(energy(mol), argnum = 1)(*args)
    geometry = geometry + 0.5 * forces

    if n % 5 == 0:
        print(f'n: {n}, E: {energy(mol)(*args):.8f}, Force-max: {abs(forces).max():.8f}')

##############################################################################
# Notice that after 35 steps of optimization the forces on the atomic nuclei and the gradient of the
# circuit parameter are both approaching zero and the energy of the molecule is that of the
# optimized geometry at the full-CI level: :math:`-1.1373060483` Ha. You can print the optimized
# geometry and verify that the final bond length of hydrogen is identical to the one computed with
# full-CI which is :math:`1.3888` `Bohr <https://en.wikipedia.org/wiki/Bohr_radius>`_.
#
# We are now ready to perform a full optimization where the circuit parameter, the atomic
# coordinates and the basis set parameters are all differentiable parameters that can be optimized
# simultaneously.

# initial values of the atomic coordinates
geometry = np.array([[0.0, 0.0, -0.672943567415407],
                     [0.0, 0.0, 0.672943567415407]], requires_grad=True)

# initial values of the basis set exponents
alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
                  [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)

# initial values of the basis set contraction coefficients
coeff = np.array([[0.1543289673, 0.5353281423, 0.4446345422],
                  [0.1543289673, 0.5353281423, 0.4446345422]], requires_grad=True)

# initial value of the circuit parameter
circuit_param = [np.array([0.0], requires_grad=True)]

for n in range(36):

    args = [circuit_param, geometry, alpha, coeff]
    mol = qml.hf.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

    # gradient for circuit parameters
    g_param = autograd.grad(energy(mol), argnum=0)(*args)
    circuit_param = circuit_param - 0.25 * g_param[0]

    # gradient for nuclear coordinates
    forces = -autograd.grad(energy(mol), argnum=1)(*args)
    geometry = geometry + 0.5 * forces

    # gradient for basis set exponents
    g_alpha = autograd.grad(energy(mol), argnum=2)(*args)
    alpha = alpha - 0.25 * g_alpha

    # gradient for basis set contraction coefficients
    g_coeff = autograd.grad(energy(mol), argnum=3)(*args)
    coeff = coeff - 0.25 * g_coeff

    if n % 5 == 0:
        print(f'n: {n}, E: {energy(mol)(*args):.8f}, Force-max: {abs(forces).max():.8f}')

##############################################################################
# You can also print the gradients of the circuit and basis set parameters and confirm that they are
# approaching zero. It is important to note that the computed energy of :math:`-1.14040160` Ha is
# lower than the full-CI energy, :math:`-1.1373060483` Ha, obtained with the sto-3g basis set for
# the hydrogen molecule because we have optimized the basis set parameters in our example. This
# means that we can reach a lower energy for hydrogen without increasing the basis set size which in
# principle leads to a larger number of qubits. You can visualize the bonding molecular orbital of
# hydrogen during each step of the optimization and monitor the change in the shape of the molecular
# orbital visually. Here is an example:
#
# .. figure:: /demonstrations/differentiable_HF/h2.gif
#     :width: 75%
#     :align: center
#
#     The bonding molecular orbital of hydrogen visualized during a full geometry, circuit and
#     basis set optimization.
#
# Conclusions
# -----------
# This tutorial introduces an important feature of PennyLane that allows performing fully-
# differentiable Hartree-Fock and subsequently VQE simulations. This feature provides two major
# benefits: i) All gradient computations needed for parameter optimization can be carried out
# with the elegant methods of automatic differentiation which facilitates simultaneous optimizations
# of circuit and Hamiltonian parameter in applications such as VQE molecular geometry optimizations.
# ii) By optimizing the molecular parameters such as the exponent and contraction coefficients of
# Gaussian functions of the basis set, one can reach a lower energy without increasing the number of
# basis functions. Can you think of other interesting molecular parameters that cab be optimized
# along with the atomic coordinates and basis set parameters that we optimized in this tutorial?
#
# References
# ----------
#
# .. [#arrazola2021]
#
#     Juan Miguel Arrazola, Soran Jahangiri, Alain Delgado, Jack Ceroni *et al.*, "Differentiable
#     quantum computational chemistry with PennyLane". `arXiv:2111.09967
#     <https://arxiv.org/abs/2111.09967>`__
#
# .. [#szabo1996]
#
#     Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic
#     Structure Theory". Dover Publications, 1996.
