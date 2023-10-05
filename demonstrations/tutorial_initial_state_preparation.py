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

It was 1968, the height of the Cold War, and on a routine surveillance mission the 
US submarine Scorpion goes missing. Stakes are high -- a nuclear sub is lost at sea! -- 
but despite the round-the-clock efforts of dozens of ships and aircraft, after a week 
the search is called off. The search area -- "somewhere off the Eastern Seaboard" -- 
is simply too big to comb through by brute force. 

But then a group of statisticians gets involved. Combining information from underwater 
sonar listening stations and averaging insights from a variety of experts, they are 
able to zero in on the a few most promising search quadrants -- and soon after, the sub 
is found in one of them.  

Much like searching the oceanic floor for a sunken ship, searching for the ground state 
in the gigantic Hilbert space of a typical molecule requires expert guidance -- an 
initial guess for what the state could be. In this demo, you will learn about different 
strategies for preparing such initial states, and specifically how to do that in 
PennyLane.

How do initial states affect quantum algorithms?
------------------------------------------------
From the variational quantum eigensolver (VQE) to quantum phase estimation (QPE), to even 
the recent ISQ-era algorithms like the Lin-Tong approach, many quantum algorithms for 
obtaining the ground state of a chemical system require a good initial state to be 
useful. (add three images here)

    1. In the case of VQE, as we will see later in this demo, a good initial state 
    directly translates into fewer optimization steps. 
    2. In QPE, the probability of measuring the ground-state energy is directly 
    proportional to the overlap squared $|c_0|^2$ of the initial and ground states

.. math::

    \ket{\psi_{\text{in}}} = c_0 \ket{\psi_0} + c_1 \ket{\psi_1} + ...

    3. Finally, in Lin-Tong the overlap with the ground-state affects the size of the 
    step in the cumulative distribution function, the bigger step making it easier to 
    detect the jump and thus resolve the position of the ground-state energy.

We see that in all these cases, having a high-quality initial state can seriously 
reduce the runtime of the algorithm. By high-quality, we just mean that the prepared 
state in some sense minimizes the effort of the quantum algorithm.

Where do I get good initial states?
-----------------------------------
Much like when searching for a sunken submarine, there are a lot of things you might try 
to prepare a good guess for the ground-state. 

Seeing as we are already using a quantum computer, we could turn to a quantum algorithm 
to do the job. One idea is based on the **adiabatic principle** that tells us that the 
eigenstates of Hamiltonians smoothly evolving with some parameter :math:`\lambda` 
are smoothly evolving also. If we start from a simple Hamiltonian :math:`H(\lambda=0)` 
whose ground state is easy to prepare, then evolve to the Hamiltonian 
:math:`H(\lambda=1)` of interest, we should end up with the ground state of the final 
Hamiltonian, or at least something close to it. In practice, the speed of state prep
 (how quickly we can evolve :math:`lambda`) depends on the spectral gap 
:math:`\Delta` between the ground and first excited states: and this gap turns out to 
be quite small precisely in interesting molecules. There are other ideas like 
quantum imaginary-time evolution and variational methods (like VQE itself), but they are 
similarly limited by long runtimes or the need for expensive classical optimization.

On the other hand, we could rely on traditional computational chemistry techniques to 
get us most of the way to an initial state. We could run a method like configuration 
interaction with singles and doubles (CISD), or coupled cluster (CCSD), take the result 
and implement it on the quantum computer. It won't not be the ground-state itself, but 
it will certainly be better than the computational basis state :math:`\ket{0}^{\otimes N}`.
It is this second approach that we focus on in this demo.

Importing initial states into PennyLane
---------------------------------------
The current version of PennyLane can import initial states from the following methods

    1. Configuration interaction with singles and doubles (CISD) from PySCF
        The first line of attack for state prep, CISD is unsophisticated but fast. 
        It won't be much help for strongly correlated molecules, but it is better 
        than Hartree-Fock.
    2. Coupled cluster with singles and doubles (CCSD) from PySCF
        In our implementation, we reconstruct the CCSD wavefunction to second order,
        making it a marginal improvement on the CISD approach.
    3. Density-matrix renormalization grourp (DMRG), from the Block2 library
        A powerful method based on matrix-product states, DMRG is considered state of the
        art for quantum chemistry simulations, capable of handling reasonably large 
        systems and correlated molecules.
    4. Semistochastic heat-bath configuration interaction (SHCI), from the Dice library
        A member of the selective configuration interaction family of methods, SHCI is 
        right there with DMRG in terms of accuracy and speed, and often used alongside it 
        as a cross-check. 

These methods are incredibly diverse in terms of their outputs, not always returning an 
object that can be turned into a PennyLane statevector. We have done this hard work: all 
that you need to do is run these methods and pass their outputs to PennyLane's 
`import_state`. 

Here is how to do this for CISD / CCSD methods via PySCF: we show the version based on 
the restricted Hartree-Fock orbitals, but the unrestricted versions are available too.
"""

from pyscf import gto, scf, ci
from pennylane import import_state
mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g', symmetry='d2h')
myhf = scf.RHF(mol).run()
myci = ci.CISD(myhf).run()
wf_cisd = import_state(myci, tol=1e-1)
print(wf_cisd)

# [pennylane output]

##############################################################################
# The general function `import_state` can automatically detect the input type and apply 
# the appropriate conversion protocol. It works similarly for CCSD

from pyscf import cc
mycc = cc.CCSD(myhf).run()
wf_ccsd = import_state(mycc, tol=1e-1)

##############################################################################
# The second attribute tol specifies the cutoff beyond which contributions to the 
# wavefunctions are neglected. Internally, wavefunctions are stored in their Slater 
# determinant representation, and if their prefactor coefficient is below `tol`, those 
# determinants are dropped from the expression.

##############################################################################
# The next two examples involve running external libraries Block2 and Dice, whose 
# installation can require some care. We recommend the install guide in our internal 
# package `Overlapper` for best performance.
# 
# To obtain an initial state from a DMRG calculation using Block2, the following is sufficient 

from pyscf import mcscf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2._pyscf.ao2mo import integrals as itg
mc = mcscf.CASCI(mf, 2, 2)
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, mc.ncore, mc.ncas, g2e_symm=8)
driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore)
ket = driver.get_random_mps(tag="GS")
driver.dmrg('add some attributes here')
wavefunction = driver.get_csf_coefficients(ket)
wf_dmrg = import_state(wavefunction, tol=1e-1)

##############################################################################
# For Dice, the following does the trick

from pyscf.shciscf import shci
import numpy as np
mol = gto.M(atom=[['Li', (0, 0, 0)], ['Li', (0,0,0.71)]], basis='sto6g', symmetry="d2h")
myhf = scf.RHF(mol).run()
ncas, nelecas_a, nelecas_b = mol.nao, mol.nelectron // 2, mol.nelectron // 2
myshci = mcscf.CASCI(myhf, ncas, (nelecas_a, nelecas_b))
output_file = f"shci_output.out"
myshci.fcisolver = shci.SHCI(myhf.mol)
myshci.fcisolver.outputFile = output_file
e_tot, e_ci, ss, mo_coeff, mo_energies = myshci.kernel(verbose=5)
(dets, coeffs) = [post-process shci_output.out to get tuple of
                                dets (list of strs) and coeffs (list of floats)]
wf_shci = import_state((dets, coeffs), tol=1e-1)

# If you are interested in a library that wraps all these methods and makes it easy to 
# generate initial states from them, you might be interested in trying out Overlapper 
# package.

##############################################################################
# Let us now demonstrate how the choice of a better initial state shortens the runtime 
# of VQE for obtaining the ground-state energy of a molecule. Start by setting up the VQE
# calculation for our molecule.

dev = qml.device("default.qubit", wires=qubits)

def circuit_VQE(theta, wires, initstate):
    qml.QubitStateVector(initstate, wires=wires)
    qml.DoubleExcitation(theta, wires=wires)

@qml.qnode(dev, interface="autograd")
def cost_fn(theta):
    circuit_VQE(theta, wires=wires)
    return qml.expval(H)

stepsize = 0.4
max_iterations = 30
opt = qml.GradientDescentOptimizer(stepsize=stepsize)
theta = np.array(0.0, requires_grad=True) 

##############################################################################
# Next, run VQE with the bad basis state 

delta_E = 10
conv_tol = 1e-8
iteration = 0
while abs(delta_E) > conv_tol and iteration < max_iterations:
    theta, prev_energy = opt.step_and_cost(cost_fn, theta, initstate=hf_state)
    samples = cost_fn(theta)
    delta_E = samples - prev_energy
    print(f"theta = {theta:.5f}, prev energy = {prev_energy:.5f}, de = {delta_E:.5f}")
    iteration += 1

energy_VQE = cost_fn(theta)
theta_opt = theta

print("VQE energy: %.4f" % (energy_VQE))
print(f"Optimal parameters: {theta_opt:.5f}")
print(f"Convergence achieved in {iteration} iterations")

##############################################################################
# Now notice how the runtime is shortened to merely a handlful of iterations with the DMRG state

delta_E = 10
conv_tol = 1e-8
iteration = 0
while abs(delta_E) > conv_tol and iteration < max_iterations:
    theta, prev_energy = opt.step_and_cost(cost_fn, theta, initstate=wf_dmrg)
    samples = cost_fn(theta)
    delta_E = samples - prev_energy
    print(f"theta = {theta:.5f}, prev energy = {prev_energy:.5f}, de = {delta_E:.5f}")
    iteration += 1

energy_VQE = cost_fn(theta)
theta_opt = theta

print("VQE energy: %.4f" % (energy_VQE))
print(f"Optimal parameters: {theta_opt:.5f}")
print(f"Convergence achieved in {iteration} iterations")

##############################################################################
# We can also consider what happens when you make the molecule more correlated. Simpler 
# methods like CISD / CCSD will begin to faulter, whiler SHCI and DMRG will continue to 
# perform at a high level

### show example ###

##############################################################################
# Finally, it is straightforward to compare the initial states through overlap -- the main
# metric of success for initial states in quantum algorithms. Because in PennyLane these 
# are statevectors, computing an overlap is as easy as computing a dot product

ovlp = np.dot(wf_dmrg, wf_shci)

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
# References
# ----------
#
# .. [#surjan]
#
#     Peter R. Surjan, "Second Quantized Approach to Quantum Chemistry". Springer-Verlag, 1989.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/stepan_fomichev.txt
