r"""

Initial state preparation for quantum chemistry
===============================================

.. meta::
    :property="og:description": Understand the concept of the initial state, and learn how to prepare it with PennyLane
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_initial_state_preparation.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Stepan Fomichev — Posted: 20 October 2023. Last updated: 20 October 2023.*

A high-quality initial state can significantly reduce the runtime of many quantum algorithms. From
the variational quantum eigensolver (VQE) to quantum phase estimation (QPE) and even the recent
`intermediate-scale quantum (ISQ) <https://pennylane.ai/blog/2023/06/from-nisq-to-isq/>`_ algorithms, obtaining the ground state of a chemical system requires
a good initial state. For instance, in the case of VQE, a good initial state directly translates into fewer
optimization steps. In QPE, the probability of measuring the ground-state energy is directly
proportional to the overlap squared of the initial and ground states. Even beyond quantum phase estimation,
good initial guesses are important for algorithms like quantum approximate optimization (QAOA)
and Grover search.

Much like searching for a needle in a haystack, there are a lot of things you might try 
to prepare a good guess for the ground state in the large-dimensional Hilbert space. In this
tutorial, we show how to use traditional computational chemistry techniques to
get us *most of the way* to an initial state. Such an initial state will not be the
ground state, but it will certainly be better than the standard guess of a computational 
basis state :math:`\ket{0}^{\otimes N}` or the Hartree-Fock state.

Importing initial states
------------------------
We can import initial states obtained from several post-Hartree-Fock quantum chemistry calculations
to PennyLane. These methods are incredibly diverse in terms of their outputs, not always returning
an object that can be turned into a PennyLane state vector. We have already done this hard
conversion work: all that you need to do is run these methods and pass their outputs
to PennyLane's :func:`~.pennylane.qchem.import_state` function. The currently supported methods are
configuration interaction with singles and doubles (CISD), coupled cluster (CCSD), density-matrix
renormalization group (DMRG) and semistochastic heat-bath configuration interaction (SHCI).

CISD states
^^^^^^^^^^^
The first line of attack for initial state preparation are CISD calculations performed with the `PySCF <https://github.com/pyscf/pyscf>`_
library. CISD is unsophisticated, but fast. It will not be of much help for strongly correlated molecules,
but it is better than Hartree-Fock. Here is the code example based on the restricted Hartree-Fock
orbitals, but the unrestricted version is available too.
"""

from pyscf import gto, scf, ci
from pennylane.qchem import import_state

R = 1.2
# create the H3+ molecule
mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,R)], ['H', (0,0,2*R)]],\
                            charge=1, basis='sto-3g')
# perfrom restricted Hartree-Fock and then CISD
myhf = scf.RHF(mol).run()
myci = ci.CISD(myhf).run()
wf_cisd = import_state(myci, tol=1e-1)
print(f"CISD-based state vector\n{wf_cisd}")

##############################################################################
# The final object, PennyLane's state vector ``wf_cisd``, is ready to be used as an 
# initial state in a quantum circuit in PennyLane--we will showcase this below for VQE.
# Conversion for CISD is straightforward: simply assign the PySCF-stored CI coefficients 
# to appropriate determinants.
#
# The second attribute, ``tol``, specifies the cutoff beyond which contributions to the 
# wavefunctions are neglected. Internally, wavefunctions are stored in their Slater 
# determinant representation, and if their prefactor coefficient is below ``tol``, those 
# determinants are dropped from the expression.
#
#
# CCSD states
# ^^^^^^^^^^^
# The function :func:`~.pennylane.qchem.import_state` is general, and can automatically detect the input type
# and apply the appropriate conversion protocol. It works similarly to the above for CCSD.

from pyscf import cc
mycc = cc.CCSD(myhf).run()
wf_ccsd = import_state(mycc, tol=1e-1)
print(f"CCSD-based state vector\n{wf_ccsd}")

##############################################################################
# For CCSD conversion, the exponential form is expanded and terms are collected to 
# second order to obtain the CI coefficients. 
#
# DMRG states
# ^^^^^^^^^^^
# The DMRG calculations involve running the library `Block2 <https://github.com/block-hczhai/block2-preview>`_, 
# which is installed from ``pip``
#
# .. code-block:: bash
#
#    pip install block2
#
# The DMRG calculation is run on top of the molecular orbitals obtained by Hartree-Fock,
# stored in the ``myhf`` object, which we can reuse from before.
#
# .. code-block:: python
#
#    from pyscf import mcscf
#    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
#    from pyblock2._pyscf.ao2mo import integrals as itg
#
#    # obtain molecular integrals and other parameters for DMRG
#    mc = mcscf.CASCI(myhf, mol.nao, mol.nelectron)
#    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = \
#                    itg.get_rhf_integrals(myhf, mc.ncore, mc.ncas, g2e_symm=8)
#
#    # initialize the DMRG solver, Hamiltonian (as matrix-product operator, MPO) and 
#    # state (as matrix-product state, MPS)
#    driver = DMRGDriver(scratch="./dmrg_temp", symm_type=SymmetryTypes.SZ)
#    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
#    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
#    ket = driver.get_random_mps(tag="GS")
#
#    # execute DMRG by modifying the ket state in-place to minimize the energy
#    driver.dmrg(mpo, ket, n_sweeps=30,bond_dims=[100,200],\
#                    noises=[1e-3,1e-5],thrds=[1e-6,1e-7],tol=1e-6)

#    # post-process the MPS to get an initial state
#    dets, coeffs = driver.get_csf_coefficients(ket, iprint=0)
#    dets = dets.tolist()
#    wf_dmrg = import_state((dets, coeffs), tol=1e-1)
#    print(f"DMRG-based state vector\n{wf_dmrg}")
#
# .. note::
#
#       DMRG-based state vector
#       [ 0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#       -0.22425623+0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.9745302 +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#        0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j]
#
# The crucial part is calling ``get_csf_coefficients()`` on the solution stored in 
# MPS form in the ``ket``. This triggers an internal reconstruction calculation that
# converts the MPS to the sum of Slater determinants form, returning the output 
# as a tuple ``(list([int]), array(float]))``. The first element expresses a given Slater
# determinant using Fock occupation vectors of length equal to the number of spatial
# orbitals in Block2 notation, where ``0`` is unoccupied, ``1`` is occupied with spin-up 
# electron, ``2`` is occupied with spin-down, and ``3`` is doubly occupied. The first 
# element must be converted to ``list`` for ``import_state`` to accept it. The second 
# element stores the CI coefficients. 
#
# In principle, this functionality can be used to generate any initial state, provided 
# the user specifies a list of Slater determinants and their coefficients in this form. 
# Let's take this opportunity to create the Hartree-Fock initial state, to compare the 
# other states against it.

from pennylane import numpy as np
hf_primer = ( [ [3, 0, 0] ], np.array([1.]) )
wf_hf = import_state(hf_primer)

##############################################################################
# SHCI states
# ^^^^^^^^^^^
# The SHCI calculations utilize the library `Dice <https://github.com/sanshar/Dice>`_, and can be run 
# using PySCF through the interface module `SHCI-SCF <https://github.com/pyscf/shciscf>`_.
# For Dice, the installation process is more complicated than for Block2, but the execution process is similar:
#
# .. code-block:: python
#
#    from pyscf.shciscf import shci
#
#    # prepare PySCF CASCI object, whose solver will be the SHCI method
#    ncas, nelecas_a, nelecas_b = mol.nao, mol.nelectron // 2, mol.nelectron // 2
#    myshci = mcscf.CASCI(myhf, ncas, (nelecas_a, nelecas_b))
#
#    # set up essentials for the SHCI solver
#    output_file = f"shci_output.out"
#    myshci.fcisolver = shci.SHCI(myhf.mol)
#    myshci.fcisolver.outputFile = output_file
#
#    # execute SHCI through the PySCF interface
#    e_tot, e_ci, ss, mo_coeff, mo_energies = myshci.kernel(verbose=5)
#
#    # post-process the result to get an initial state
#    (dets, coeffs) = [post-process shci_output.out to get tuple of
#                                  dets (list([str])) and coeffs (array([float]))]
#    wf_shci = import_state((dets, coeffs), tol=1e-1)
#    print(f"SHCI-based state vector\n{wf_shci}")
#
# .. note::
#    
#    SHCI-based state vector
#    [ 0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.22425623+0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#     -0.97453022+0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
#      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j]
#
# If you are interested in a library that wraps all these methods and makes it easy to 
# generate initial states from them, you should try Overlapper, our internal 
# package built specifically for using traditional quantum chemistry methods 
# to construct initial states.
#
##############################################################################
# Application: speed up VQE
# -------------------------
#
# Let us now demonstrate how the choice of a better initial state shortens the runtime 
# of VQE for obtaining the ground-state energy of a molecule. As a first step, create a
# molecule, a device, and a simple VQE circuit with double excitations

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

# generate the molecular Hamiltonian for H3+
H2mol, qubits = qchem.molecular_hamiltonian(["H", "H", "H"],\
                        np.array([0,0,0,0,0,R/0.529, 0,0,2*R/0.529]),\
                            charge=1,basis="sto-3g")
wires = list(range(qubits))
dev = qml.device("default.qubit", wires=qubits)

# create all possible excitations in H3+
singles, doubles = qchem.excitations(2, qubits)
excitations = singles + doubles

##############################################################################
# Now let's run VQE with the Hartree-Fock initial state

# VQE circuit with wf_hf as initial state and all possible excitations
@qml.qnode(dev, interface="autograd")
def circuit_VQE(theta):
    qml.StatePrep(wf_hf, wires=wires)
    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(theta[i], wires=excitation)
        else:
            qml.SingleExcitation(theta[i], wires=excitation)
    return qml.expval(H2mol)

# create the VQE optimizer, initialize the variational parameters, set start params
opt = qml.GradientDescentOptimizer(stepsize=0.4)
theta = np.array(np.zeros(len(excitations)), requires_grad=True)
delta_E, iteration = 10, 0
results_hf = []

# run the VQE optimization loop until convergence threshold is reached
while abs(delta_E) > 1e-5:
    theta, prev_energy = opt.step_and_cost(circuit_VQE, theta)
    new_energy = circuit_VQE(theta)
    delta_E = new_energy - prev_energy
    results_hf.append(new_energy)
print(f"Starting with HF state took {len(results_hf)} iterations until convergence.")

##############################################################################
# And compare with how things go when you run it with the CISD initial state

# re-create VQE circuit with wf_cisd as initial state
@qml.qnode(dev, interface="autograd")
def circuit_VQE(theta):
    qml.StatePrep(wf_cisd, wires=wires)
    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(theta[i], wires=excitation)
        else:
            qml.SingleExcitation(theta[i], wires=excitation)
    return qml.expval(H2mol)

theta = np.array(np.zeros(len(excitations)), requires_grad=True)
delta_E, iteration = 10, 0
results_cisd = []

while abs(delta_E) > 1e-5:
    theta, prev_energy = opt.step_and_cost(circuit_VQE, theta)
    new_energy = circuit_VQE(theta)
    delta_E = new_energy - prev_energy
    results_cisd.append(new_energy)
print(f"Starting with CISD state took {len(results_cisd)} iterations until convergence.")

# plot the results
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(range(len(results_hf)), results_hf, color="r", marker="o", label="HF")
ax.plot(range(len(results_cisd)), results_cisd, color="b", marker="o", label="CISD")
ax.legend(fontsize=16)
ax.tick_params(axis="both", labelsize=16)
ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel("Energy, Ha", fontsize=20)
plt.tight_layout()
plt.show()

##############################################################################
# Finally, it is straightforward to compare the initial states through overlap--a traditional
# metric of success for initial states in quantum algorithms. Because in PennyLane these 
# are regular arrays, computing an overlap is as easy as computing a dot product

print(np.dot(wf_cisd, wf_hf))
print(np.dot(wf_ccsd, wf_hf))
#
# .. code-block:: python
#    print(np.dot(wf_dmrg, wf_hf))
#    print(np.dot(wf_shci, wf_hf))
#
# .. note:: 
#    (0.9745302156335056+0j)
#    (-0.9745302156443371+0j)
#
##############################################################################
# In this particular case, even CISD gives the exact wavefunction, hence all overlaps 
# are identical. In more correlated molecules, overlaps will show that the more 
# multireference methods DMRG and SHCI are farther away from the Hartree-Fock state, 
# allowing them to perform better. If a ground state in such a case was known, the 
# overlap to it could tell us directly the quality of the initial state.

##############################################################################
# Summary
# -------
# This demo explains the concept of the initial state for quantum algorithms. Using the 
# example of VQE, it demonstrates how a better choice of state--obtained, for example 
# from a sophisticated computational chemistry method like CCSD, SHCI or DMRG--can lead
# to much better algorithmic performance. It also shows simple workflows for how to run 
# these computational chemistry methods, from libraries such as `PySCF <https://github.com/pyscf/pyscf>`_, 
# `Block2 <https://github.com/block-hczhai/block2-preview>`_ and 
# `Dice <https://github.com/sanshar/Dice>`_, to generate outputs that can then be 
# converted to PennyLane's state vector format with a single line of code.
#
# About the author
# ----------------
# .. include:: ../_static/authors/stepan_fomichev.txt
