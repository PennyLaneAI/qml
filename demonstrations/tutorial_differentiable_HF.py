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
parametrized quantum circuit and find a set of parameters that minimize the expectation value of the
electronic molecular Hamiltonian. For a given molecule, the electronic Hamiltonian is obtained by
solving the Hartree-Fock calculations which provide a set of one- and two-electron integrals that
are used to construct the Hamiltonian. The constructed Hamiltonian depends on a set of molecular
parameters, such as the atomic coordinates and basis set parameters, which are excluded from the
optimization problem. The ability to optimize these molecular parameters concurrently with the
circuit parameters provides several computational advantages such as efficient molecular geometry
optimization and reaching lower energies without increasing the number of basis functions.

Optimization of the molecular Hamiltonian parameters, at the same time as the circuit parameters,
can be achieved by differentiating the expectation value of the Hamiltonian with respect to the
molecular parameters which can be done with symbolic, numeric, or automatic differentiation.
Symbolic differentiation obtains derivatives of an input function by direct mathematical
manipulation, for example using standard strategies of differential calculus. These can be performed
by hand or with the help of computer algebra software. The resulting expressions are exact, but the
symbolic approach is of limited scope, particularly since many functions are not known in explicit
analytical form. Symbolic methods also suffer from the expression swell problem where careless usage
can lead to exponentially large symbolic expressions. Numerical differentiation is a versatile but
unstable method, often relying on finite differences to calculate approximate derivatives. This is
problematic especially for large computations consisting of many differentiable parameters.
Automatic differentiation is a computational strategy where a function implemented using computer
code is differentiated by expressing it in terms of elementary operations for which derivatives are
known. The gradient of the target function is then obtained by applying the chain rule through the
entire code. In principle, automatic differentiation can be used to calculate derivatives of a
function using resources comparable to those required to evaluate the function itself.

In this tutorial, you will learn how to use the autodifferentiable Hartree-Fock solver implemented
in PennyLane. The Hartree-Fock module in PennyLane provides built-in methods for constructing atomic
and molecular orbitals, building Fock matrices, and solving the self-consistent Hartree-Fock
equations to obtain optimized orbitals, which can be used to construct fully-differentiable
molecular Hamiltonians. PennyLane allows users to natively compute derivatives of all these objects
with respect to the underlying parameters. We will introduce a workflow to jointly optimize circuit
parameters, nuclear coordinates, and basis set parameters in a variational quantum eigensolver
algorithm. Let's get started!

Hartree-Fock method
-------------------

The main goal of the Hartree-Fock method is to obtain molecular spin-orbitals that minimize the
energy of a state where electrons are treated as independent particles occupying the lowest-energy
orbitals. These optimized molecular orbitals are then used to construct one- and two-body electron
integrals in the molecular orbital basis which are used to generate differentiable second-quantized
Hamiltonians in the fermionic and qubit basis.

To get started, we need to define the atomic symbols and coordinates of the desired molecule. For
the hydrogen molecule we define the symbols and geometry as
"""

import autograd
import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H"]
geometry = np.array([[0.0, 0.0, -0.672943567415407],
                     [0.0, 0.0,  0.672943567415407]], requires_grad=True)

##############################################################################
# Note that `requires_grad=True` specifies that the atomic coordinates are differentiable
# parameters. We can now compute the Hartree-Fock energy and its gradient with respect to the atomic
# coordinates for hydrogen. To do that, we need to create a molecule object which stores all the
# molecular parameters needed to perform a Hartree-Fock calculation.

mol = qml.hf.Molecule(symbols, geometry)

##############################################################################
# The Hartree-Fock energy can now be computed with the :func:`~.pennylane.hf.hf_energy` function

qml.hf.hf_energy(mol)(geometry)

##############################################################################
# The computed energy matches the reference value of :math:`-1.1175058849` Ha. We now compute the
# gradient of the energy with respect to the atomic coordinates with autograd

autograd.grad(qml.hf.hf_energy(mol))(geometry)

##############################################################################
# Not that we need to pass the `mol` object and the values of the atomic coordinates. The computed
# gradients are equal or very close to zero because the initial geometry we used here has been
# already optimized at the Hartree-Fock level.

# We can also compute the values and gradients of several other quantities that are computed during
# the Hartree-Fock procedure. These include all integrals over basis functions, matrices formed from
# these integrals and the one- and two-body integrals over molecular orbitals. Let's look at few
# examples.

# We first compute the overlap integral between the two S atomic orbitals located on each of the
# hydrogen atoms. Remember that in the 3to-3g basis set each of the atomic orbitals is represented
# by a single basis function which is stored in the molecule object.

S1 = mol.basis_set[0]
S2 = mol.basis_set[1]

##############################################################################
# We can check the parameters of the basis functions as

S1.params

# which returns the exponents, contraction coeeficients and centers of the three Gaussian functions
# of the sto-3g basis set. These data can be obtained individually by using `S1.alpha`, `S1.coeff`
# and `S1.r` respectively. You can verify that both of the S1 and S2 orbitals have the same
# exponents and contraction coefficients but are centered on different hydrogen atoms. You can also
# verify that the orbitals are S-type ones by returning the angular momentum quantum numbers with
# S1.l which gives you a tuple of three integers, representing the exponents of the x, y and z
# components in the Gaussian functions.

# Having the two atomic orbitals, we can now compute the overlap integral by passing the orbitals
# and their centers to the :func:`~.pennylane.hf.generate_overlap` function. Note that the centers
# of the orbitals are those of the hydrogen atoms by default and are therefore treated as
# differentiable parameters by PennyLane.

qml.hf.generate_overlap(S1, S2)([S1.r, S2.r])

##############################################################################
# You can verify that the overlap integral between two identical atomic orbitals is zero. We can now
# compute the gradient of the overlap integral with respect to the Gaussian centers

autograd.grad(qml.hf.generate_overlap(S1, S2))([S1.r, S2.r])

##############################################################################
# Can you explain why some of the computed gradients are zero?

# Let's now compute the molecular Hamiltonian with the :func:`~.pennylane.hf.generate_hamiltonian`

hamiltonian = qml.hf.generate_hamiltonian(mol)(geometry)
print(hamiltonian)

##############################################################################
# The Hamiltonian contains 15 terms and importantly the coefficients of the Hamiltonian are also
# differentiable. We can construct a circuit and perform a VQE simulation in which both of the
# circuit parameters and the molecular geometries are optimized simultaneously by using gradients
# computed with the methods of automatic differentiation.

dev = qml.device("default.qubit", wires=4)
def energy(mol):
    @qml.qnode(dev)
    def circuit(*args):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
        qml.DoubleExcitation(*args[0][0], wires=[0, 1, 2, 3])
        return qml.expval(qml.hf.generate_hamiltonian(mol)(*args[1:]))
    return circuit

##############################################################################
# We now compute the gradients of the energy with respect to the circuit parameters and the atomic
# coordinates. Note that the latter gradients are simply the forces on the atomic nuclai.

circuit_param = [np.array([0.0], requires_grad=True)]

for n in range(50):

    args = [circuit_param, geometry]
    mol = qml.hf.Molecule(symbols, geometry)

    # gradient for circuit parameters
    g_param = autograd.grad(energy(mol), argnum = 0)(*args)
    circuit_param = circuit_param - 0.25 * g_param[0]

    # gradient for nuclear coordinates
    forces = -autograd.grad(energy(mol), argnum = 1)(*args)
    geometry = geometry + 0.5 * forces

    if n % 5 == 0:
        print(f'n: {n}, E: {energy(mol)(*args):.8f}, Force-max: {abs(forces).max():.8f}, G-param: {abs(g_param[0][0]):.8f}')

##############################################################################
# Notice that after 50 steps of optimization the forces on the atomic nuclai and the gradient of the
# circuit parameter are both approaching zero and the energy og the molecule is that of the
# optimized geometry at the full-CI level: :math:`-1.1373060483` Ha. You can print the optimized
# geometry and verify that the final bond length of hydrogen is that of the full-CI which is
# :math:`1.3888` Bohr.