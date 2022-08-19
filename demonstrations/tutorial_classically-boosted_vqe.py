r"""
Classically-Boosted Variational Quantum Eigensolver
===================================================

.. meta::
    :property="og:description": Learn how to implement classically-boosted VQE in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/CB_VQE.png

.. related::
    
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe Variational Quantum Eigensolver

*Authors: Joana Fraxanet & Isidor Schoch (Xanadu Residents).
Posted: 27 July 2022. Last updated: 27 July 2022.*

One of the most important applications of quantum computers is expected
to be the computation of ground-state energies of complicated molecules
and materials. Even though there are already some solid proposals on how
to tackle these problems when fault-tolerant quantum computation comes
into play, we currently live in the `noisy intermediate-scale quantum 
(NISQ)<https://en.wikipedia.org/wiki/Noisy_intermediate-scale_quantum_era>`__
era, meaning that we can only access noisy and limited devices.
That is why a large part of the current research on quantum algorithms is
focusing on what can be done with few resources. In particular, most
proposals rely on variational quantum algorithms (VQA), which are
optimized classically and adapt to the limitations of the quantum
devices. For the specific problem of computing ground-state energies,
the paradigmatic algorithm is the Variational Quantum Eigensolver
(VQE) algorithm.

.. figure:: ../demonstrations/classically_boosted-vqe/quantum_algorithms.png
    :align: center
    :width: 50%

Although VQE is intended to run on NISQ devices, it is nonetheless
sensitive to noise. This is particularly problematic when applying a
large number of gates. As a consequence, several modifications to the
original VQE algorithm have been proposed. These variants are usually
intended to improve the algorithm’s performance on NISQ-era devices.

Here we will go through one of these proposals step-by-step: the
Classically-Boosted Variational Quantum Eigensolver (CB-VQE) [#Radin2021]_. 
Implementing CB-VQE reduces the number of measurements required to obtain the
ground-state energy with a certain precision. This is done by making use
of classical states which already contain some information about the
ground-state of the problem.

.. figure:: ../demonstrations/classically_boosted-vqe/CB_VQE.png
    :align: center
    :width: 50%

We will restrict ourselves to the :math:`H_2` molecule for
the sake of simplicity. First, we will give a short introduction on how
to perform standard VQE for the molecule of interest. For more details,
we recommend
`this brief overview of VQE <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__ to learn
how to implement VQE for molecules step by step. Then, we will implement
the CB-VQE algorithm for the specific case in which we rely only on one
classical state⁠—that being the Hartree-Fock state. Finally, we will
discuss the number of measurements needed to obtain a certain
error-threshold by comparing the two methods.

Let’s get started!

"""

######################################################################
# Prequisites: Standard VQE
# -------------------------
# 
# If you are not already familiar with the VQE family of algorithms and
# wish to see how one can apply it to the :math:`H_2` molecule, feel free to
# work through the aforementioned demo before reading this section. 
# Here, we will only briefly review the main idea behind standard VQE 
# and highlight the important concepts in connection with CB-VQE.
# 
# Given an Hamiltonian :math:`H`, the main goal of VQE is to find the ground energy of the Schrödinger
# equation
# 
# .. math:: H \vert \phi \rangle = E  \vert \phi \rangle.
# 
# This corresponds to the problem of diagonalizing the Hamiltonian and
# finding the smallest eigenvalue. Alternatively, one can formulate the
# problem using the variational principle, in which we are interested in
# minimizing the energy
# 
# .. math:: E = \langle \phi \vert H \vert \phi \rangle.
# 
# In VQE, we prepare a statevector :math:`\vert \phi \rangle` by applying
# the parameterized ansatz :math:`A(\Theta)`, represented by a unitary matrix, 
# to an inital state :math:`\vert 0^N \rangle`. Then, the parameters :math:`\Theta` are
# optimized to minimize a cost function, which in this case is the energy:
# 
# .. math::  E(\Theta) = \langle 0 \vert^{\otimes N} A(\Theta)^{\dagger} H A(\Theta) \vert 0 \rangle^{\otimes N}. 
# 
# This is done using a classical optimization method, which is typically
# gradient descent.
# 
# To implement our example of VQE, we first define the molecular
# Hamiltonian for the :math:`H_2` molecule in the minimal *STO-3G basis*
# using PennyLane
# 

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
basis_set = "sto-3g"
electrons = 2

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    basis=basis_set,
)


######################################################################
# We then initialize the Hartree-Fock state
# :math:`\vert \phi_{HF}\rangle=\vert 1100 \rangle`
# 

hf = qml.qchem.hf_state(electrons, qubits)


######################################################################
# and implement the ansatz :math:`A(\Theta)`. In this case, we use the
# `AllSinglesDoubles <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.AllSinglesDoubles.html?highlight=allsinglesdoubles#pennylane.AllSinglesDoubles>`__
# class, which enables us to apply all possible combinations of single and
# double excitations obeying the Pauli principle to the Hartree-Fock
# state. Single and double excitation gates :math:`G^{(1, 2)}(\Theta)` are
# conveniently implemented in PennyLane with
# `SingleExcitation <https://pennylane.readthedocs.io/en/latest/code/api/pennylane.SingleExcitation.html>`__
# and
# `DoubleExcitation <https://pennylane.readthedocs.io/en/latest/code/api/pennylane.DoubleExcitation.html>`__
# classes.
# 

singles, doubles = qml.qchem.excitations(electrons=electrons, orbitals=qubits)
num_theta = len(singles) + len(doubles)

def circuit_VQE(theta, wires):
    qml.AllSinglesDoubles(
        weights = theta,
        wires = wires,
        hf_state = hf,
        singles = singles,
        doubles = doubles)


######################################################################
# Once this is defined, we can run the VQE algorithm. We first need to
# define a circuit for the cost function. For our purposes of studying the
# performance of VQE with the number of measurements, we will take a
# finite number of shots ``num_shots = 100.``
# 

num_shots = 100
dev = qml.device('default.qubit', wires=qubits, shots=int(num_shots))
@qml.qnode(dev)
def cost_fn(theta):
    circuit_VQE(theta,range(qubits))
    return qml.expval(H)


######################################################################
# we then fix the classical optimization parameters ``stepsize`` and
# ``max_iteration``
# 

stepsize = 0.4
max_iterations = 10
opt = qml.GradientDescentOptimizer(stepsize=stepsize)
theta = np.zeros(num_theta, requires_grad=True)


######################################################################
# Finally, we run the algorithm
# 

for n in range(max_iterations):

    theta, prev_energy = opt.step_and_cost(cost_fn, theta)
    samples = cost_fn(theta)
           
energy_VQE = cost_fn(theta)
theta_opt = theta

print('VQE for num. of shots: %.0f \nEnergy: %.4f' %(num_shots, energy_VQE))


######################################################################
# Note that as an output we obtain the VQE approximation to the ground
# state energy and a set of optimized parameters :math:`\Theta` that
# define the ground state through the ansatz :math:`A(\Theta)`. We will need
# to save these two quantities, as they are necessary to implement CB-VQE
# in the following steps.
# 


######################################################################
# Classically-Boosted VQE
# -----------------------
# 
# Now we are ready to present the classically-boosted version of VQE.
# 
# The key of this new method relies on the notion of
# `generalized eigenvalue problem <https://en.wikipedia.org/wiki/Generalized_eigenvalue_problem>`__.
# The main idea is to restrict the problem of finding the ground state to
# an eigenvalue problem in a subspace :math:`\mathcal{H}^{\prime}` of the
# complete Hilbert space :math:`\mathcal{H}`. If this subspace is spanned
# by a combination of both classical and quantum states, we can run parts
# of our algorithm on classical hardware and thus reduce the number of
# measurements needed to reach a certain precision threshold. For a
# subspace spanned by the states
# :math:`\{\vert \phi_\alpha \rangle\}_{\alpha\in \mathcal{H}^{\prime}}`,
# the generalized eigenvalue problem is expressed as
# 
# .. math:: \bar{H} \vec{v}=  \lambda \bar{S} \vec{v},
# 
# where :math:`\bar{H}` is the Hamiltonian :math:`H` projected into the
# subspace of interest, i.e. with the entries
# 
# .. math:: \bar{H}_{\alpha, \beta} = \langle \phi_\alpha \vert H \vert \phi_\beta \rangle \quad \forall \alpha, \beta \in \mathcal{H}^{\prime} ,
# 
# and the matrix :math:`\bar{S}` contains the overlaps between the basis
# states. For a complete orthonormal basis, the overlap matrix
# :math:`\bar{S}` would simply be the identity matrix. However, we need to
# take a more general approach which works for a subspace spanned by
# potentially non-orthogonal states. We can retrieve the representation of
# :math:`S` in terms of :math:`\{\vert \phi_\alpha \rangle\}_\alpha` by
# calculating
# 
# .. math:: \bar{S}_{\alpha, \beta} = \langle \phi_\alpha \vert \phi_\beta \rangle \quad \forall \alpha, \beta \in \mathcal{H}^{\prime}.
# 
# Note that :math:`\vec{v}` and :math:`\lambda` are the eigenvectors and
# eigenvalues respectively. In particular, our goal is to find the lowest
# eigenvalue :math:`\lambda_0.`
# 


######################################################################
# Equipped with the useful mathematical description of generalized
# eigenvalue problems, we can now choose our subspace such that some of
# the states :math:`\phi_{\alpha} \in \mathcal{H}^{\prime}` are
# classically tractable.
# 
# We will consider the simplest case in which the subspace is spanned only
# by one classical state :math:`\vert \phi_{HF} \rangle` and one quantum
# state :math:`\vert \phi_{q} \rangle`. More precisely, we define the
# classical state to be a single
# `Slater determinant <https://en.wikipedia.org/wiki/Slater_determinant>`__,
# which directly hints towards using the *Hartree-Fock* state for several
# reasons. First of all, it is well-known that the Hartree-Fock state is a
# good candidate to approximate the ground state in the mean-field limit.
# Secondly, we already computed it when we built the molecular Hamiltonian
# for the standard VQE!
# 


######################################################################
# To summarize, our goal is to build the Hamiltonian :math:`\bar{H}` and
# the overlap matrix :math:`\bar{S}`, which act on the subspace
# :math:`\mathcal{H}^{\prime} \subseteq \mathcal{H}` spanned by
# :math:`\{\vert \phi_{HF} \rangle, \vert \phi_q \rangle\}`. These will be
# two-dimensional matrices, and in the following sections we will show how
# to compute all their entries step by step.
# 
# As done previously, we start by importing *PennyLane*, *Qchem* and
# differentiable *NumPy* followed by defining the molecular Hamiltonian in
# the Hartree-Fock basis for :math:`H_2`.
# 

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

# Define the molecular Hamiltonian
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
basis_set = "sto-3g"

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    basis=basis_set,
)


######################################################################
# Computing Classical Quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# We first set out to calculate the purely classical part of the
# Hamiltonian :math:`H`. Since we only have one classical state this will
# already correspond to a scalar energy value. The terms can be expressed
# as
# 
# .. math:: H_{11} = \langle \phi_{HF} \vert H \vert \phi_{HF} \rangle \quad \text{and} \quad S_{11} = \langle \phi_{HF} \vert \phi_{HF} \rangle
# 
# which is tractable using classical methods. This energy corresponds to
# the Hatree-Fock energy due to our convenient choice of the classical
# state. Note that the computation of the classical compononent of the
# overlap matrix
# :math:`S_{11} = \langle \phi_{HF} \vert \phi_{HF} \rangle = 1` is
# trivial.
# 
# Using PennyLane, we can access the Hartree-Fock energy by looking at the
# fermionic Hamiltonian, which is the Hamiltonian in the basis of Slater
# determinants. The basis is organized in lexicographic order, meaning
# that if we want the entry corresponding to the Hartree-Fock determinant
# :math:`\vert 1100 \rangle`, we will have to take the entry
# :math:`H_{i,i}`, where :math:`1100` is the binary representation of the
# index :math:`i`.
# 

hf_state = qml.qchem.hf_state(electrons, qubits)
fermionic_Hamiltonian = qml.utils.sparse_hamiltonian(H).toarray()

binary_string = ''.join([str(i) for i in hf_state])
idx0 = int(binary_string, 2)
H11 = fermionic_Hamiltonian[idx0][idx0]
S11 = 1


######################################################################
# Computing Quatum Quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# We now move on to the purely quantum part of the Hamiltonian, i.e. the
# entry
# 
# .. math:: H_{22} = \langle \phi_{q} \vert H \vert \phi_{q} \rangle,
# 
# where :math:`\vert \phi_q \rangle` is the quantum state. This state is
# just the output of the standard VQE with a given ansatz, following the
# steps in the first section. Therefore, the entry :math:`H_{22}` just
# corresponds to the final energy of the VQE. In particular, note that the
# quantum state can be written as
# :math:`\vert \phi_{q} \rangle = A(\Theta^*) \vert \phi_{HF} \rangle`
# where :math:`A(\Theta^*)` is the ansatz of the VQE with the optimised
# parameters :math:`\Theta^*`. Once again, we have
# :math:`S_{22}=\langle \phi_{q} \vert \phi_{q} \rangle = 1` for the
# overlap matrix.
# 

H22 = energy_VQE
S22 = 1


######################################################################
# Computing Mixed Quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# The final part of the algorithm computes the cross-terms between the
# classical and quantum state
# 
# .. math:: H_{12} = \langle \phi_{HF} \vert H \vert \phi_{q} \rangle = H_{21}^{\dagger}.
# 
# This part of the algorithm is slightly more complicated than the
# previous steps, since we still want make use of the classical component
# of the problem in order to minimize the number of required shots.
# 
# Keep in mind that most algorithms usually perform computations either on
# fully classically or quantum tractable Hilbert spaces. CB-VQE takes
# advantage of the classical part of the problem while still calculating a
# classically-intractable quantity by using the so-called
# `Hadamard test <https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)>`__
# to construct :math:`H_{12}`. The Hadamard test is a prime example of an
# indirect measurement, which allows us to measure properties of a state
# without (completely) destroying it.
# 
# As the Hadamard test returns the real part of a coefficient from a unitary representing 
# an operation, we will focus on calculating the quantities
# 
# .. math:: H_{12} = \sum_{i} Re(\langle \phi_q \vert i \rangle) \langle i \vert H \vert \phi_{HF} \rangle,
# 
# .. math:: S_{12} = Re(\langle \phi_q \vert \phi_{HF} \rangle),
# 
# where :math:`\lvert i \rangle` are the computational basis states of the system,
# i.e. the basis of single Slater determinants. Note that we have to decompose the Hamiltonian 
# into a sum of unitaries. For the problem under consideration, the set of relevant computational basis states for which
# :math:`\langle i \vert H \vert \phi_{HF}\rangle \neq 0` contains all the
# single and double excitations (allowed by spin symmteries)
# 
# .. math:: \vert 1100 \rangle, \vert 1001 \rangle, \vert 0110 \rangle, \vert 0011 \rangle.
# 
# Note that the set of computational basis states includes the
# *Hartree-Fock* state :math:`\lvert i_0 \rangle = \phi_{HF} = \vert 1100 \rangle`. The
# projections :math:`\langle i \vert H \vert \phi_{HF} \rangle` can be
# extracted analytically from the fermionic Hamiltonian that we computed
# above, by accessing the entries by the index given by the binary
# expression of each Slater determinant.
# 
# The Hadamard test is required in order to compute the real part of
# :math:`\langle \phi_q \vert i \rangle`.
# 
# .. figure:: ../demonstrations/classically_boosted-vqe/hadamard_test.png
#     :align: center
#     :width: 50%
#
# To implement the Hadamard test, we need a register of :math:`N` qubits
# given by the size of the molecular Hamiltonian (:math:`N=4` in our case)
# initialized in the state :math:`\rvert 0^N \rangle` and an ancillary
# qubit prepared in the :math:`\rvert 0 \rangle` state.
# 
# In order to generate :math:`\langle \phi_q \vert i \rangle`, we take
# :math:`U_q` such that
# :math:`U_q \vert 0^N \rangle = \vert \phi_q \rangle`. In particular,
# this is equivalent to using the standard VQE ansatz with the optimized
# parameters :math:`\Theta*` that we obtained in the previous section
# :math:`U_q = A(\Theta*)` applied on the *Hartree-Fock* state. Moreover,
# we also need :math:`U_i` such that
# :math:`U_i \vert 0^N \rangle = \vert \phi_i \rangle`. In this case, this
# is just a mapping of a classical basis state into the circuit consisting
# of :math:`X` gates and can be easily implemented using PennyLane’s
# function ``qml.BasisState(i, N))``.
# 

num_shots = 100
wires = range(qubits + 1)
dev = qml.device("default.qubit", wires=wires, shots=num_shots)

@qml.qnode(dev)
def hadamard_test(Uq, Ucl, component='real'):

    if component == 'imag':
        qml.RX(math.pi/2, wires=wires[1:])

    qml.Hadamard(wires=[0])
    qml.ControlledQubitUnitary(Uq.conjugate().T @ Ucl, control_wires=[0], wires=wires[1:])
    qml.Hadamard(wires=[0])

    return qml.probs(wires=[0])


######################################################################
# Now, we can compute the Hamiltonian
# cross-terms.
# 

def circuit_product_state(state):
    qml.BasisState(state, range(qubits))

Uq = qml.matrix(circuit_VQE)(theta_opt, range(qubits)) @ qml.matrix(circuit_product_state)([1,1,0,0])

H12 = 0
relevant_basis_states = np.array([[1,1,0,0], [0,1,1,0], [1,0,0,1], [0,0,1,1]], requires_grad=True)
for j, basis_state in enumerate(relevant_basis_states):
    Ucl = qml.matrix(circuit_product_state)(basis_state)
    probs = hadamard_test(Uq, Ucl)
    y = 2*abs(probs[0])-1
    binary_string = ''.join([str(coeff) for coeff in basis_state])
    idx = int(binary_string, 2)
    overlap_H = fermionic_Hamiltonian[idx0][idx]
    H12 += y * overlap_H
    if j == 0:
        y0 = y
        
H21 = np.conjugate(H12)


######################################################################
# Finally, we can define the cross terms of the :math:`S` matrix making
# use of the projections with the *Hartree-Fock* state.
# 

S12 = y0
S21 = y0.conjugate()


######################################################################
# Solving the generalized eigenvalue problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Finally, we are ready to solve the generalized eigenvalue problem. For
# this, we will build the matrices :math:`H` and :math:`S` and use scipy
# to obtain the lowest eigenvalue.
# 

from scipy import linalg

S = np.array([[S11, S12],[S21, S22]])
H = np.array([[H11, H12],[H21, H22]])

evals = linalg.eigvals(H, S)
energy_CBVQE = np.min(evals).real

print('CB-VQE for num. of shots %.0f \nEnergy %.4f' %(num_shots, energy_CBVQE))


######################################################################
# Measurement analysis
# --------------------
# 


######################################################################
# CB-VQE is helpful when it comes to reducing the number of measurements
# that are required to reach a given precision in the ground state energy.
# For very small systems it can be shown that the classically-boosted method 
# reduces the number of required measurements by a factor of :math:`1000` [#Radin2021]_.
# 
# For this demo, we run the standard VQE and CB-VQE algorihtms :math:`100`
# times for different values of ``num_shots``. We then compute the mean
# value of the energies and the standard deviation for both cases.
# 
#
# .. figure:: ../demonstrations/classically_boosted-vqe/energy_deviation.png
#     :align: center
#     :width: 80%
#
# We see that CB-VQE leads to lower energies improving the results given
# by standard VQE. For the limit of large ``num_shots`` we see that, as
# expected, both algorithms converge to the same value of the ground state
# energy.
#
# # References
# ----------
#
# .. [#Radin2021]
#
#     M. D. Radin. (2021) "Classically-Boosted Variational Quantum Eigensolver",
#     (`arXiv <https://arxiv.org/abs/2106.04755>`__) 

