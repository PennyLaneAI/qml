r"""

Initial state preparation for quantum chemistry
===============================================

A high-quality initial state can significantly reduce the runtime of many quantum algorithms. From
the variational quantum eigensolver (VQE) to quantum phase estimation (QPE) and even the recent
`intermediate-scale quantum (ISQ) <https://pennylane.ai/blog/2023/06/from-nisq-to-isq/>`_ algorithms, obtaining the ground state of a chemical system requires
a good initial state. For instance, in the case of VQE, a good initial state directly translates into fewer
optimization steps. In QPE, the probability of measuring the ground-state energy is directly
proportional to the overlap squared of the initial and ground states. Even beyond quantum phase estimation,
good initial guesses are important for algorithms like quantum approximate optimization (QAOA)
and Grover search.

Much like searching for a needle in a haystack, there are a lot of things you might try 
to prepare a good guess for the ground state in the high-dimensional Hilbert space. In this
tutorial, we show how to use traditional computational chemistry techniques to
get us *most of the way* to an initial state. Such an initial state will not be the
ground state, but it will certainly be better than the standard guess of a computational 
basis state :math:`\ket{0}^{\otimes N}` or the Hartree-Fock state.

.. figure:: ../demonstrations/initial_state_preparation/qchem_input_states.png
    :align: center
    :width: 65%
    :target: javascript:void(0)

Importing initial states
------------------------
We can import initial states obtained from several post-Hartree-Fock quantum chemistry methods
to PennyLane. These methods are incredibly diverse in terms of their outputs, not always returning
an object that is easy to turn into a PennyLane state vector.

We have already done this hard conversion work: all that you need to do is run these methods and
pass their outputs to PennyLane's :func:`~.pennylane.qchem.import_state` function. The currently 
supported methods are configuration interaction with singles and doubles (CISD), coupled cluster 
(CCSD), density-matrix renormalization group (DMRG) and semistochastic heat-bath configuration 
interaction (SHCI).

We now show how this works on :math:`\text{H}_3^+` molecule as an example.


CISD states
~~~~~~~~~~~
The first line of attack for initial state preparation is often a CISD calculation, performed with the `PySCF <https://github.com/pyscf/pyscf>`_
library. CISD is unsophisticated, but fast. It will not be of much help for strongly correlated molecules,
but it is better than Hartree-Fock. Here is the code example using the restricted Hartree-Fock
orbitals (it also works for unrestricted orbitals too).
"""

from pyscf import gto, scf, ci
from pennylane.qchem import import_state
from pennylane import numpy as np

R = 1.2
# create the H3+ molecule
mol = gto.M(atom=[["H", (0, 0, 0)],
                  ["H", (0, 0, R)],
                  ["H", (0, 0, 2 * R)]], charge=1)
# perfrom restricted Hartree-Fock and then CISD
myhf = scf.RHF(mol).run()
myci = ci.CISD(myhf).run()
wf_cisd = import_state(myci, tol=1e-1)
print(f"CISD-based state vector: \n {np.round(wf_cisd.real, 4)}")

##############################################################################
# The final object, PennyLane's state vector ``wf_cisd``, is ready to be used as an
# initial state in a quantum circuit in PennyLane--we will showcase this below for VQE.

# Conversion for CISD to a state vector is straightforward: simply assign the PySCF-stored 
# CI coefficients to appropriate determinants.
# The second attribute passed to ``import_state()``, ``tol``, specifies the cutoff beyond
# which contributions to the wavefunctions are neglected. Internally, wavefunctions are 
# stored in their Slater determinant representation. If their prefactor coefficient
# is below ``tol``, those determinants are dropped from the expression.
#
#
# CCSD states
# ~~~~~~~~~~~
# The function :func:`~.pennylane.qchem.import_state` is general and works similarly for CCSD. It can 
# automatically detect the input type and apply the appropriate conversion protocol. 

from pyscf import cc

mycc = cc.CCSD(myhf).run()
wf_ccsd = import_state(mycc, tol=1e-1)
print(f"CCSD-based state vector: \n {np.round(wf_ccsd.real, 4)}")

##############################################################################
# For CCSD conversion, the exponential form is expanded and terms are collected **to 
# second order** to obtain the CI coefficients. 
#
# DMRG states
# ~~~~~~~~~~~
# For more complex or more correlated molecules, initial states from DMRG or
# SHCI will be better options. DMRG calculations involve running the library `Block2 <https://github.com/block-hczhai/block2-preview>`_,
# which can be installed with ``pip``:
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
#
#    # post-process the MPS to get an initial state
#    dets, coeffs = driver.get_csf_coefficients(ket, iprint=0)
#    dets = dets.tolist()
#    wf_dmrg = import_state((dets, coeffs), tol=1e-1)
#    print(f"DMRG-based state vector: \n {np.round(wf_dmrg, 4)}")
#
# .. code-block:: bash
#
#    DMRG-based state vector
#     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.     -0.2243  0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.9745  0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.    ]

##############################################################################
# The crucial part is calling ``get_csf_coefficients()`` on the solution stored in
# MPS form in the ``ket``. This triggers an internal reconstruction calculation that
# converts the MPS to the sum of Slater determinants form, returning the output 
# as a tuple ``(array([int]), array(float]))``. The first element expresses a given Slater
# determinant using Fock occupation vectors of length equal to the number of spatial
# orbitals in Block2 notation, where ``0`` is unoccupied, ``1`` is occupied with spin-up
# electron, ``2`` is occupied with spin-down, and ``3`` is doubly occupied. The first
# element must be converted to ``list`` for :func:`~.pennylane.qchem.import_state` to accept it.
# The second element stores the CI coefficients.
#
# In principle, this functionality can be used to generate any initial state, provided
# the user specifies a list of Slater determinants and their coefficients in this form.
# Let's take this opportunity to create the Hartree-Fock initial state, to compare the
# other states against it.

from pennylane import numpy as np

hf_primer = ([[3, 0, 0]], np.array([1.0]))
wf_hf = import_state(hf_primer)

##############################################################################
# SHCI states
# ~~~~~~~~~~~
#
# The SHCI calculations utilize the library `Dice <https://github.com/sanshar/Dice>`_, and can be run
# using PySCF through the interface module `SHCI-SCF <https://github.com/pyscf/shciscf>`_.
# For Dice, the execution process is similar:
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
#    # post-process the shci_output.out to extract the wave function 
#    # results and create the tuple of dets (list([str])) and coeffs (array([float]))
#    # shci_data = (dets, coeffs)
#    wf_shci = import_state(shci_data, tol=1e-1)
#    print(f"SHCI-based state vector\n{wf_shci}")
#
# .. code-block:: bash
#
#    SHCI-based state vector
#     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.2243  0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.     -0.9745  0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.    ]

##############################################################################
# The Dice output file prints determinants using symbols ``0`` (unoccupied orbital), 
# ``a`` and ``b`` (orbital occupied with spin-up and spin-down electron, respectively),
# and ``2`` (doubly occupied orbital). These determinant outputs, and corresponding 
# coefficients, should be extracted and arranged as ``(list([str]), array(float]))``,
# where each string combines all the determinant symbols ``0, a, b, 2`` for a single 
# determinant.

##############################################################################
# Application: speed up VQE
# -------------------------
# Let us now demonstrate how the choice of a better initial state shortens the runtime
# of VQE for obtaining the ground-state energy of a molecule. As a first step, create a
# molecule, a device, and a simple VQE circuit with double excitations:

import pennylane as qml
from pennylane import qchem

# generate the molecular Hamiltonian for H3+
symbols = ["H", "H", "H"]
geometry = np.array([[0, 0, 0], [0, 0, R/0.529], [0, 0, 2*R/0.529]])

H2mol, qubits = qchem.molecular_hamiltonian(symbols, geometry, charge=1)
wires = list(range(qubits))
dev = qml.device("default.qubit", wires=qubits)

# create all possible excitations in H3+
singles, doubles = qchem.excitations(2, qubits)
excitations = singles + doubles

##############################################################################
# Now let's run VQE with the Hartree-Fock initial state:


@qml.qnode(dev)
def circuit_VQE(theta, initial_state):
    qml.StatePrep(initial_state, wires=wires)
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
    theta, prev_energy = opt.step_and_cost(circuit_VQE, theta, initial_state=wf_hf)
    new_energy = circuit_VQE(theta, initial_state=wf_hf)
    delta_E = new_energy - prev_energy
    results_hf.append(new_energy)
    if len(results_hf) % 5 == 0:
        print(f"Step = {len(results_hf)},  Energy = {new_energy:.6f} Ha")
print(f"Starting with HF state took {len(results_hf)} iterations until convergence.")

##############################################################################
# And compare with how things go when you run it with the CISD initial state:

theta = np.array(np.zeros(len(excitations)), requires_grad=True)
delta_E, iteration = 10, 0
results_cisd = []

while abs(delta_E) > 1e-5:
    theta, prev_energy = opt.step_and_cost(circuit_VQE, theta, initial_state=wf_cisd)
    new_energy = circuit_VQE(theta, initial_state=wf_cisd)
    delta_E = new_energy - prev_energy
    results_cisd.append(new_energy)
    if len(results_cisd) % 5 == 0:
        print(f"Step = {len(results_cisd)},  Energy = {new_energy:.6f} Ha")
print(f"Starting with CISD state took {len(results_cisd)} iterations until convergence.")

##############################################################################
# Let's visualize the comparison between the two initial states, and see that indeed 
# we get to the ground state much faster by starting with the CISD state.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(len(results_hf)), results_hf, color="r", marker="o", label="wf_hf")
ax.plot(range(len(results_cisd)), results_cisd, color="b", marker="o", label="wf_cisd")
ax.legend(fontsize=16)
ax.tick_params(axis="both", labelsize=16)
ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel("Energy, Ha", fontsize=20)
plt.tight_layout()
plt.show()

##############################################################################
# Finally, it is straightforward to compare the initial states through overlap--a traditional
# metric of success for initial states in quantum algorithms. Because in our examples these
# are regular arrays, computing an overlap is as easy as computing a dot product

print(np.dot(wf_cisd, wf_hf).real)
print(np.dot(wf_ccsd, wf_hf).real)

##############################################################################
# In this particular case, even CISD gives the exact wavefunction, hence all overlaps
# are identical. In more correlated molecules, overlaps will show that the more
# multireference methods DMRG and SHCI are farther away from the Hartree-Fock state,
# allowing them to perform better (you can check this by printing the overlaps with 
# DMRG and SHCI in a more correlated molecule). If a ground state in such a case was known, 
# the overlap to it could tell us directly the quality of the initial state.

##############################################################################
# Conclusion
# -----------
# This demo shows how to import initial states from outputs of traditional quantum chemistry methods 
# for use in PennyLane-based algorithms. We showcased simple workflows for how to run 
# a variety of state-of-the-art post-Hartree-Fock methods, from libraries such as 
# `PySCF <https://github.com/pyscf/pyscf>`_, 
# `Block2 <https://github.com/block-hczhai/block2-preview>`_ and
# `Dice <https://github.com/sanshar/Dice>`_, to generate outputs that can then be
# converted to PennyLane's state vector format with a single line of code. With these 
# initial states, we use the example of VQE to demonstrate how a better choice 
# of initial state can lead to improved algorithmic performance. For the molecule 
# used in our example, the CISD state was sufficient: however, in more correlated 
# molecules, DMRG and SHCI initial states typically provide the best speed-ups.
#
# About the author
# ----------------
# .. include:: ../_static/authors/stepan_fomichev.txt
