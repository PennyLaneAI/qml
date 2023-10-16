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
the variational quantum eigensolver (VQE) to quantum phase estimation (QPE), to even the recent
intermediate-scale quantum (ISQ) algorithms, obtaining the ground state of a chemical system requires
a good initial state. For instance, in the case of VQE, a good initial state directly translates into fewer
optimization steps. In QPE, the probability of measuring the ground-state energy is directly
proportional to the overlap squared of the initial and ground states. Even beyond quantum phase estimation,
good initial guesses are important for algorithms like quantum approximate optimization (QAOA)
and Grover search.

Much like searching for a needle in a haystack, there are a lot of things you might try 
to prepare a good guess for the ground-state in the large-dimensional Hilbert space. In this
tutorial, we show how to use traditional computational chemistry techniques to
get us *most of the way* to an initial state. Such an initial state will not be the
ground-state, but it will certainly be better than the standard guess of a computational 
basis state :math:`\ket{0}^{\otimes N}` or the Hartree-Fock state.

Importing initial states
------------------------
We can import initial states obtained from several post-Hartree-Fock quantum chemistry calculations
to PennyLane. These methods are incredibly diverse in terms of their outputs, not always returning
an object that can be turned into a PennyLane statevector. We have already done this hard
work of conversion: all that you need to do is run these methods and pass their outputs
to PennyLane's :func:`~.pennylane.qchem.import_state` function. The currently supported methods are
configuration interaction with singles and doubles (CISD), coupled cluster (CCSD), density-matrix
renormalization group (DMRG) and semistochastic heat-bath configuration interaction (SHCI).

CISD states
^^^^^^^^^^^
The first line of attack for initial state preparation are CISD calculations performed with the `PySCF <https://github.com/pyscf/pyscf>`_
library. CISD is unsophisticated, but fast. It will not be much help for strongly correlated molecules,
but it is better than Hartree-Fock. Here is the code example based on the restricted Hartree-Fock
orbitals, but the unrestricted version is available too.
"""

from pyscf import gto, scf, ci
from pennylane.qchem import import_state
R = 0.71
mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,R)]], basis='sto6g', symmetry='d2h')
myhf = scf.RHF(mol).run()
myci = ci.CISD(myhf).run()
wf_cisd = import_state(myci, tol=1e-1)
print(f"CISD-based statevector\n{wf_cisd}")

##############################################################################
# The final object, PennyLane's statevector ``wf_cisd``, is ready to be used as an 
# initial state in a quantum circuit in PennyLane -- we will showcase this below for VQE.
# Conversion for CISD is straightforward: simply assign the PySCF-stored CI coefficients 
# to appropriate determinants.
#
# CCSD states
# ^^^^^^^^^^^
# The function :func:`~.pennylane.qchem.import_state` is general, and can automatically detect the input type
# and apply the appropriate conversion protocol. It works similarly to the above for CCSD.

from pyscf import cc
mycc = cc.CCSD(myhf).run()
wf_ccsd = import_state(mycc, tol=1e-1)
print(f"CCSD-based statevector\n{wf_ccsd}")

##############################################################################
# For CCSD conversion, the exponential form is expanded and terms are collected to 
# second order to obtain the CI coefficients. 
#
# The second attribute ``tol`` specifies the cutoff beyond which contributions to the 
# wavefunctions are neglected. Internally, wavefunctions are stored in their Slater 
# determinant representation, and if their prefactor coefficient is below ``tol``, those 
# determinants are dropped from the expression.
#
# DMRG states
# ^^^^^^^^^^^
# The DMRG calculations involve running the library `Block2 <https://github.com/block-hczhai/block2-preview>`_. 
# Block2 installs simply from ``pip``
#
# .. code-block:: bash
#
#    pip install block2
#
# The DMRG calculation is run on top of the molecular orbitals obtained by Hartree-Fock,
# stored in ``myhf`` object, which we can re-use from before.
#
# .. code-block::python
#
#    from pyscf import mcscf
#    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
#    from pyblock2._pyscf.ao2mo import integrals as itg
#    mc = mcscf.CASCI(myhf, 2, 2)
#    ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = \
#                    itg.get_rhf_integrals(myhf, mc.ncore, mc.ncas, g2e_symm=8)
#    driver = DMRGDriver(scratch="./dmrg_temp", symm_type=SymmetryTypes.SZ)
#    driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
#    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
#    ket = driver.get_random_mps(tag="GS")
#    driver.dmrg(mpo, ket, n_sweeps=30,bond_dims=[100,200],\
#                    noises=[1e-3,1e-5],thrds=[1e-6,1e-7],tol=1e-6)
#    dets, coeffs = driver.get_csf_coefficients(ket, iprint=0)
#    dets = dets.tolist()
#    wf_dmrg = import_state((dets, coeffs), tol=1e-1)
#    print(f"DMRG-based statevector\n{wf_dmrg}")
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
#
# SHCI states
# ^^^^^^^^^^^
# The SHCI calculations involve running the library Dice. For Dice, the installation process is more
# complicated but the execution process is similar:
#
# .. code-block:: bash
#
#    from pyscf.shciscf import shci
#    ncas, nelecas_a, nelecas_b = mol.nao, mol.nelectron // 2, mol.nelectron // 2
#    myshci = mcscf.CASCI(myhf, ncas, (nelecas_a, nelecas_b))
#    output_file = f"shci_output.out"
#    myshci.fcisolver = shci.SHCI(myhf.mol)
#    myshci.fcisolver.outputFile = output_file
#    e_tot, e_ci, ss, mo_coeff, mo_energies =
#    myshci.kernel(verbose=5)
#    wavefunction = get_dets_coeffs_output(output_file)
#    print(type(wavefunction[0][0]))
#    print(dets, coeffs)
#    (dets, coeffs) = [post-process shci_output.out to get tuple of
#                                  dets (list of strs) and coeffs (list of floats)]
#    wf_shci = import_state((dets, coeffs), tol=1e-1)
#    print(f"SHCI-based statevector\n{wf_shci}")
#
# If you are interested in a library that wraps all these methods and makes it easy to 
# generate initial states from them, you should try Overlapper, our internal 
# package built specifically for using traditional quantum chemistry methods 
# to construct initial states.
#
# Application: speed up VQE
# -------------------------
#
# Let us now demonstrate how the choice of a better initial state shortens the runtime 
# of VQE for obtaining the ground-state energy of a molecule. As a first step, create a
# molecule, a device and a simple VQE circuit with double excitations

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
H2mol, qubits = qchem.molecular_hamiltonian(["H", "H"],\
                        np.array([0,0,0,0,0,R/0.529]),basis="sto-3g")

dev = qml.device("default.qubit", wires=qubits)

def circuit_VQE(theta, wires, initstate):
    qml.StatePrep(initstate, wires=wires)
    qml.DoubleExcitation(theta, wires=wires)

@qml.qnode(dev, interface="autograd")
def cost_fn(theta, initstate=None, ham=H2mol):
    circuit_VQE(theta, wires=list(range(qubits)), initstate=initstate)
    return qml.expval(ham)

##############################################################################
# The ``initstate`` variable is where we can insert different initial states. Next, create a
# function to execute VQE

def run_VQE(initstate, ham=H2mol, conv_tol=1e-4, max_iterations=30):
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    delta_E, iteration = 10, 0
    while abs(delta_E) > conv_tol and iteration < max_iterations:
        theta, prev_energy = opt.step_and_cost(cost_fn, theta, initstate=initstate, ham=ham)
        new_energy = cost_fn(theta, initstate=initstate, ham=ham)
        delta_E = new_energy - prev_energy
        iteration += 1
    energy_VQE = cost_fn(theta, initstate=initstate, ham=ham)
    theta_opt = theta
    return energy_VQE, theta_opt

##############################################################################
# Now let's compare the number of iterations to convergence for the Hartree-Fock state 
# versus the CCSD state

wf_hf = np.zeros(2**qubits)
wf_hf[3] = 1.
energy_hf, theta_hf = run_VQE(wf_hf)
energy_ccsd, theta_ccsd = run_VQE(wf_ccsd)

##############################################################################
# We can also consider what happens when you make the molecule more correlated, for example
# by stretching its bonds. Simpler methods like HF will require even more VQE iterations, 
# while SHCI and DMRG will continue to provide good starting points for the algorithm.

H2mol_corr, qubits = qchem.molecular_hamiltonian(["H", "H"],\
                        np.array([0,0,0,0,0,R*2/0.529]),basis="sto-3g")
energy_hf, theta_hf = run_VQE(wf_hf, ham=H2mol_corr)
energy_ccsd, theta_ccsd = run_VQE(wf_ccsd, ham=H2mol_corr)
# energy_dmrg, theta_dmrg = run_VQE(wf_dmrg, ham=H2mol_corr)

##############################################################################
# Finally, it is straightforward to compare the initial states through overlap -- a traditional
# metric of success for initial states in quantum algorithms. Because in PennyLane these 
# are statevectors, computing an overlap is as easy as computing a dot product

ovlp = np.dot(wf_ccsd, wf_hf)

##############################################################################
# Summary
# -------
# This demo explains the concept of the initial state for quantum algorithms. Using the 
# example of VQE, it demonstrates how a better choice of state -- obtained, for example 
# from a sophisticated computational chemistry method like CCSD, SHCI or DMRG -- can lead
# to much better algorithmic performance. It also shows simple workflows for how to run 
# these computational chemistry methods, from libraries such as PySCF, Block2 and Dice, 
# to generate outputs that can then be converted to PennyLane's statevector format 
# with a single line of code.
#
# About the author
# ----------------
# .. include:: ../_static/authors/stepan_fomichev.txt
