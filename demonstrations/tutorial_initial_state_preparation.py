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

How do initial states affect quantum algorithms?
------------------------------------------------
From the variational quantum eigensolver (VQE) to quantum phase estimation (QPE), to even 
the recent ISQ-era algorithms like the Lin-Tong approach, many quantum algorithms for 
obtaining the ground state of a chemical system require a good initial state to be 
useful. (add three images here)

In the case of VQE, as we will see later in this demo, a good initial state directly 
translates into fewer optimization steps. In QPE, for an initial state 
:math:`\ket{\psi_{\text{in}}}` written in terms of the eigenstates $\{\ket{\psi_n}\}$ 
of the system Hamiltonian

.. math::

    \ket{\psi_{\text{in}}} = c_0 \ket{\psi_0} + c_1 \ket{\psi_1} + ...

the probability of measuring the ground-state energy is directly proportional to the 
overlap squared $|c_0|^2$ of the initial and ground states. Finally, in Lin-Tong, the 
overlap with the ground-state affects the size of the step in the cumulative 
distribution function. A bigger step makes it easier to detect the jump with fewer 
circuit samples, and thus resolve the position of the ground-state energy. Even beyond 
quantum phase estimation, good initial guesses are important for algorithms like 
quantum approximate optimization (QAOA) and Grover search.

To summarize, having a high-quality initial state can seriously reduce the runtime 
of many quantum algorithms. By high-quality, we just mean that the prepared 
state in some sense minimizes the effort of the quantum algorithm.

Where to get good initial states?
-----------------------------------
Much like searching for a needle in a haystack, there are a lot of things you might try 
to prepare a good guess for the ground-state in the large-dimensional Hilbert space. 

Seeing as we are already using a quantum computer, we could turn to a quantum algorithm 
to do the job -- this is the domain of quantum heuristics. The most famous idea is the 
adiabatic state preparation approach, but there are others like quantum imaginary-time 
evolution (QITE) and variational methods (for example, VQE). Unfortunately, these 
methods are typically similarly limited by a) long runtimes, or b) the need for 
expensive classical optimization, and c) provide no performance guarantees.

On the other hand, we could rely on traditional computational chemistry techniques to 
get us _most of the way_ to an initial state. We could run a method like configuration 
interaction with singles and doubles (CISD), or coupled cluster (CCSD), take the result 
and implement it on the quantum computer. Such an initial state will not be the 
ground-state, but it will certainly be better than the standard guess of a computational 
basis state :math:`\ket{0}^{\otimes N}` or the Hartree-Fock state. 

It is this second approach that we focus on in this demo.

Importing initial states into PennyLane
---------------------------------------
The current version of PennyLane can import initial states from the following methods

    1. Configuration interaction with singles and doubles (CISD) from PySCF
        The first line of attack for state prep, CISD is unsophisticated but fast. 
        It will not be much help for strongly correlated molecules, but it is better 
        than Hartree-Fock.
    2. Coupled cluster with singles and doubles (CCSD) from PySCF
        In our implementation, we reconstruct the CCSD wavefunction to second order,
        making it a marginal improvement on the CISD approach.
    3. Density-matrix renormalization group (DMRG), from the Block2 library
        A powerful method based on matrix-product states, DMRG is considered state of the
        art for quantum chemistry simulations, capable of handling reasonably large 
        systems (100-140 spin orbitals) and strongly correlated molecules.
    4. Semistochastic heat-bath configuration interaction (SHCI), from the Dice library
        A member of the selective configuration interaction family of methods, SHCI is 
        right there with DMRG in terms of accuracy and speed, and often used alongside it 
        as a cross-check. 

These methods are incredibly diverse in terms of their outputs, not always returning an 
object that can be turned into a PennyLane statevector. We have already done this hard 
work of conversion: all that you need to do is run these methods and pass their outputs 
to PennyLane's `import_state`. 

Here is how to do this for CISD / CCSD methods via PySCF: we show the version based on 
the restricted Hartree-Fock orbitals, but the unrestricted versions are available too.
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
# The final object, PennyLane's statevector `wf_cisd`, is ready to be used as an 
# initial state in a quantum circuit in PennyLane -- we will showcase this below for VQE.
# Conversion for CISD is straightforward: simply assign the PySCF-stored CI coefficients 
# to appropriate determinants.
#
# The function `import_state` is general, and can automatically detect the input type 
# and apply the appropriate conversion protocol. It works similarly to the above for CCSD

from pyscf import cc
mycc = cc.CCSD(myhf).run()
wf_ccsd = import_state(mycc, tol=1e-1)
print(f"CCSD-based statevector\n{wf_ccsd}")

# For CCSD conversion, the exponential form is expanded and terms are collected to 
# second order to obtain the CI coefficients. 

##############################################################################
# The second attribute `tol` specifies the cutoff beyond which contributions to the 
# wavefunctions are neglected. Internally, wavefunctions are stored in their Slater 
# determinant representation, and if their prefactor coefficient is below `tol`, those 
# determinants are dropped from the expression.

##############################################################################
# The next two examples involve running external libraries Block2 and Dice, whose 
# installation can require some care. While Block2 can be installed with `pip`, for 
# Dice we recommend the install guide in our internal package `Overlapper`.
# 
# To install the Block2 library with functionality needed for this demo, execute

"""
.. code::

pip install block2==0.5.2rc10 --extra-index-url=https://block-hczhai.github.io/block2-preview/pypi/
"""  

# The DMRG calculation is run on top of the molecular orbitals obtained by Hartree-Fock,
# stored in `myhf` object, which we can re-use from before.

from pyscf import mcscf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg
mc = mcscf.CASCI(myhf, 2, 2)
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = \
                    itg.get_rhf_integrals(myhf, mc.ncore, mc.ncas, g2e_symm=8)
driver = DMRGDriver(scratch="./dmrg_temp", symm_type=SymmetryTypes.SZ)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
ket = driver.get_random_mps(tag="GS")
driver.dmrg(mpo, ket, n_sweeps=30,bond_dims=[100,200],\
                noises=[1e-3,1e-5],thrds=[1e-6,1e-7],tol=1e-6)
dets, coeffs = driver.get_csf_coefficients(ket, iprint=0)
dets = dets.tolist()
wf_dmrg = import_state((dets, coeffs), tol=1e-1)
print(f"DMRG-based statevector\n{wf_dmrg}")

# The crucial part is calling `get_csf_coefficients()` on the solution stored in 
# MPS form in the `ket`. This triggers an internal reconstruction calculation that
# converts the MPS to the sum of Slater determinants form, returning the output 
# as a tuple `(list([int]), array(float])). The first element expresses a given Slater
# determinant using Fock occupation vectors of length equal to the number of spatial
# orbitals in Block2 notation, where `0` is unoccupied, `1` is occupied with spin-up 
# electron, `2` is occupied with spin-down, and `3` is doubly occupied. The first 
# element must be converted to `list` for `import_state` to accept it. The second 
# element stores the CI coefficients. 
#
# In principle, this functionality can be used to generate any initial state, provided 
# the user specifies a list of Slater determinants and their coefficients in this form. 

##############################################################################
# For Dice, the installation process is more complicated (see the Overlapper install 
# guide), but the execution process is similar:

# .. note::
#
#   .. code-block:: python
#        >>> from pyscf.shciscf import shci
#        >>> ncas, nelecas_a, nelecas_b = mol.nao, mol.nelectron // 2, mol.nelectron // 2
#        >>> myshci = mcscf.CASCI(myhf, ncas, (nelecas_a, nelecas_b))
#        >>> output_file = f"shci_output.out"
#        >>> myshci.fcisolver = shci.SHCI(myhf.mol)
#        >>> myshci.fcisolver.outputFile = output_file
#        >>> e_tot, e_ci, ss, mo_coeff, mo_energies = 
#        >>> myshci.kernel(verbose=5)
#        >>> wavefunction = get_dets_coeffs_output(output_file)
#        >>> print(type(wavefunction[0][0]))
#        >>> print(dets, coeffs)
#        >>> (dets, coeffs) = [post-process shci_output.out to get tuple of
#        >>>                                 dets (list of strs) and coeffs (list of floats)]
#        >>> wf_shci = import_state((dets, coeffs), tol=1e-1)
#        >>> print(f"SHCI-based statevector\n{wf_shci}")

# If you are interested in a library that wraps all these methods and makes it easy to 
# generate initial states from them, you should try Overlapper, our internal 
# package built specifically for using traditional quantum chemistry methods 
# to construct initial states.

##############################################################################
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
    qml.QubitStateVector(initstate, wires=wires)
    qml.DoubleExcitation(theta, wires=wires)

@qml.qnode(dev, interface="autograd")
def cost_fn(theta, initstate=None, ham=H2mol):
    circuit_VQE(theta, wires=list(range(qubits)), initstate=initstate)
    return qml.expval(ham)

# The `initstate` variable is where we can insert different initial states.

##############################################################################
# Next, create a function to execute VQE

def run_VQE(initstate, ham=H2mol, conv_tol=1e-4, max_iterations=30):
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    delta_E, iteration = 10, 0
    while abs(delta_E) > conv_tol and iteration < max_iterations:
        theta, prev_energy = opt.step_and_cost(cost_fn, theta, initstate=initstate, ham=ham)
        new_energy = cost_fn(theta, initstate=initstate, ham=ham)
        delta_E = new_energy - prev_energy
        print(f"theta = {theta:.5f}, prev energy = {prev_energy:.5f}, de = {delta_E:.5f}")
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

"""
.. figure:: ../demonstrations/initial_state/hf_vs_ccsd_on_vqe.png
   :scale: 65%
   :alt: Comparing HF and CCSD initial states for VQE on H2 molecule.
   :align: center
"""

##############################################################################
# We can also consider what happens when you make the molecule more correlated. Simpler 
# methods like HF will begin to faulter, whiler SHCI and DMRG will continue to 
# perform at a high level

H2mol_corr, qubits = qchem.molecular_hamiltonian(["H", "H"],\
                        np.array([0,0,0,0,0,R*2/0.529]),basis="sto-3g")
energy_hf, theta_hf = run_VQE(wf_hf, ham=H2mol_corr)
energy_ccsd, theta_ccsd = run_VQE(wf_ccsd, ham=H2mol_corr)
energy_dmrg, theta_dmrg = run_VQE(wf_dmrg, ham=H2mol_corr)

"""
.. figure:: ../demonstrations/initial_state/hf_vs_ccsd_on_vqe_stretched.png
   :scale: 65%
   :alt: Comparing HF and CCSD initial states for VQE on stretched H2 molecule.
   :align: center
"""

##############################################################################
# Finally, it is straightforward to compare the initial states through overlap -- the main
# metric of success for initial states in quantum algorithms. Because in PennyLane these 
# are statevectors, computing an overlap is as easy as computing a dot product

ovlp = np.dot(wf_dmrg, wf_hf)

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
