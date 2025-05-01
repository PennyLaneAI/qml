r"""Density Matrix Embedding Theory (DMET)
=========================================
Materials simulation presents a crucial challenge in quantum chemistry. Density Functional Theory
(DFT) is currently the workhorse for simulating materials due to its balance between accuracy and
computational efficiency. However, it often falls short in accurately capturing the intricate
electron correlation effects found in strongly correlated materials. As a result, researchers often
turn to more sophisticated methods, such as full configuration interaction or coupled cluster
theory, which provide better accuracy but come at a significantly higher computational cost.

Embedding theories provide a balanced midpoint solution that enhances our ability to simulate
materials accurately and efficiently. The core idea behind embedding is that the system is divided
 into two parts: an impurity which is a strongly correlated subsystem that requires exact
description and an environment which can be treated with approximate but computationally efficient
method.

Density matrix embedding theory (DMET) is an efficient embedding approach to treat strongly
correlated systems. Here we provide a brief introduction of the method and demonstrate how to run
DMET calculations to construct a Hamiltonian that can be used in a quantum algorithm.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""
# Theory
# ------
# The wave function for the embedded system composed of the impurity and the environment can be
# simply represented as
#
# .. math::
#
#      | \Psi \rangle = \sum_i^{N_I} \sum_j^{N_E} \Psi_{ij} | I_i \rangle | E_j \rangle,
#
# where :math:`I_i` and :math:`E_j` are basis states of the impurity :math:`I` and environment
# :math:`E`, respectively, :math:`\Psi_{ij}` is the matrix of coefficients and :math:`N` is the
# number of sites, e.g., orbitals. The key idea in DMET is to perform a singular value decomposition
# of the coefficient matrix :math:`\Psi_{ij} = \sum_{\alpha} U_{i \alpha} \lambda_{\alpha} V_{\alpha j}`
# and rearrange the wave functions such that
#
# .. math::
#
#      | \Psi \rangle = \sum_{\alpha}^{N} \lambda_{\alpha} | A_{\alpha} \rangle | B_{\alpha} \rangle,
#
# where :math:`A_{\alpha} = \sum_i U_{i \alpha} | I_i \rangle` are states obtained from rotations of
# :math:`I_i` to a new basis and :math:`B_{\alpha} = \sum_j V_{j \alpha} | E_j \rangle` are bath
# states representing the portion of the environment that interacts with the impurity. Note that the
# number of bath states is identical by the number of fragment states, regardless of the size of the
# environment. This new decomposition is the Schmidt decomposition of the system wave function.
#
# We are now able to project the full Hamiltonian to the space of impurity and bath states, known as
# embedding space.
#
# .. math::
#
#      \hat{H}^{emb} = \hat{P}^{\dagger} \hat{H}^{sys}\hat{P}
#
# where :math:`P = \sum_{\alpha \beta} | A_{\alpha} B_{\beta} \rangle \langle A_{\alpha} B_{\beta}|`
# is a projection operator.
#
# Note that the Schmidt decomposition requires apriori knowledge of the wavefunction. To alleviate
# this, DMET operates through a systematic iterative approach, starting with a meanfield description
# of the wavefunction and refining it through feedback from solution of the impurity Hamiltonian.
#
# Implementation
# --------------
# The DMET procedure starts by getting an approximate description of the system. This approximate
# wavefunction is then partitioned with Schmidt decomposition to get the impurity and bath orbitals
# which are used to define an approximate projector :math:`P`. The projector is then used to
# construct the embedded Hamiltonian. This Hamiltonian is then solved using accurate methods such as
# post-Hartree-Fock methods, exact diagonalisation, or accurate quantum algorithms. the results are
# used to re-construct the projector and the process is repeated until the wave function converges.
# Let's now take a look at the implementation of these steps for the $H_6$ system. We use the
# programs PySCF [#pyscf]_ and libDMET which can be installed with
#
# Constructing the system
# ^^^^^^^^^^^^^^^^^^^^^^^
# We begin by defining a hydrogen chain with 6 atoms using PySCF. This is done by creating
# a ``Cell`` object with three unit cell each containing two Hydrogen atoms at a bond distance of
# 0.75 Å. Then, we construct a ``Lattice`` object from the libDMET library, associating it with
# the defined cell.
#
# .. code-block:: python
#
#    import numpy as np
#    from pyscf.pbc import gto, df, scf, tools
#    from libdmet.system import lattice
#
#    cell = gto.Cell()
#    cell.a = ''' 10.0    0.0     0.0
#                 0.0     10.0    0.0
#                 0.0     0.0     1.5 ''' # lattice vectors for unit cell
#    cell.atom = ''' H 0.0      0.0      0.0
#                    H 0.0      0.0      0.75 ''' # coordinates of atoms in unit cell
#    cell.basis = '321g'
#    cell.build(unit='Angstrom')
#
#    kmesh = [1, 1, 3] # number of k-points in xyz direction
#
#    lat = lattice.Lattice(cell, kmesh)
#    filling = cell.nelectron / (lat.nscsites * 2.0)
#    kpts = lat.kpts
#
# Performing mean-field calculations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can now perform a mean-field calculation on the whole system through Hartree-Fock with density
# fitted integrals using PySCF.
#
# .. code-block:: python
#
#    gdf = df.GDF(cell, kpts)
#    gdf._cderi_to_save = 'gdf_ints.h5' # output file for density fitted integral tensor
#    gdf.build() # compute the density fitted integrals
#
#    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
#    kmf.with_df = gdf # use density-fitted integrals
#    kmf.with_df._cderi = 'gdf_ints.h5' # input file for density fitted integrals
#    kmf.kernel() # run Hartree-Fock
#
# Partitioning the orbital space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we have an approximate description of our system and can start obtaining the impurity and bath
# orbitals. This requires the localization of the basis of orbitals. We can use any localized basis
# such as molecular orbitals (MO) or intrinsic atomic orbitals (IAO) [#SWouters]_. The use of
# localized basis provides a convenient way to understand the contribution of each atom to
# properties of the full system. Here, we rotate the one-electron and two-electron integrals into
# IAO basis.
#
# .. code-block:: python
#
#    from libdmet.basis_transform import make_basis
#
#    c_ao_iao, _, _, lo_labels = make_basis.get_C_ao_lo_iao(lat, kmf, minao="MINAO", full_return=True, return_labels=True)
#    c_ao_lo = lat.symmetrize_lo(c_ao_iao)
#    lat.set_Ham(kmf, gdf, c_ao_lo, eri_symmetry=4) # rotate integral tensors to IAO basis
#
# We now obtain the orbital labels for each atom in the unit cell and define the impurity and bath
# by looking at the labels. In this example, we choose to keep the :math:`1s` orbitals in the unit
# cell in the impurity, while the bath contains the :math:`2s` orbitals, and the orbitals belonging
# to the rest of the supercell become part of the unentangled environment. These can be separated by
# getting the valence and virtual labels from get_labels function.
#
# .. code-block:: python
#
#    from libdmet.lo.iao import get_labels
#
#    aoind = cell.aoslice_by_atom()
#    labels, val_labels, virt_labels = get_labels(cell, minao="MINAO")
#    ncore = 0
#    lat.set_val_virt_core(len(val_labels), len(virt_labels), ncore)
#
#    print("Valence orbitals: ", val_labels)
#    print("Virtual orbitals: ", virt_labels)
#
# Self-Consistent DMET
# ^^^^^^^^^^^^^^^^^^^^
# Now that we have a description of our impurity and bath orbitals, we can implement the iterative
# process of DMET. We implement each step of the process in a function and then call these functions
# to perform the calculations. Note that if we only perform one step of the iteration the process is
# referred to as single-shot DMET.
#
# We first need to construct the impurity Hamiltonian.
#
# .. code-block:: python
#
#    def construct_impurity_hamiltonian(lat, v_cor, filling, mu, last_dmu, int_bath=True):
#
#        rho, mu, scf_result = dmet.HartreeFock(lat, v_cor, filling, mu,
#                                        ires=True, labels=lo_labels)
#        imp_ham, _, basis = dmet.ConstructImpHam(lat, rho, v_cor, int_bath=int_bath)
#        imp_ham = dmet.apply_dmu(lat, imp_ham, basis, last_dmu)
#
#        return rho, mu, scf_result, imp_ham, basis
#
# Next, we solve this impurity Hamiltonian with a high-level method. The following function defines
# the electronic structure solver for the impurity, provides an initial point for the calculation
# and passes the ``Lattice`` information to the solver. The solver then calculates the energy and
# density matrix for the impurity.
#
# .. code-block:: python
#
#    def solve_impurity_hamiltonian(lat, cell, basis, imp_ham, last_dmu, scf_result):
#
#        solver = dmet.impurity_solver.FCI(restricted=True, tol=1e-13)
#        basis_k = lat.R2k_basis(basis) #basis in k-space
#
#        solver_args = {"nelec": min((lat.ncore+lat.nval)*2, lat.nkpts*cell.nelectron), \
#                   "dm0": dmet.foldRho_k(scf_result["rho_k"], basis_k)}
#
#        rho_emb, energy_emb, imp_ham, dmu = dmet.SolveImpHam_with_fitting(lat, filling,
#            imp_ham, basis, solver, solver_args=solver_args)
#
#        last_dmu += dmu
#        return rho_emb, energy_emb, imp_ham, last_dmu, [solver, solver_args]
#
# The final step is to include the effect of the environment in the expectation value. So we define
# a function which returns the density matrix and energy for the whole system.
#
# .. code-block:: python
#
#    def solve_full_system(lat, rho_emb, energy_emb, basis, imp_ham, last_dmu, solver_info, lo_labels):
#        rho_full, energy_full, nelec_full = \
#                dmet.transformResults(rho_emb, energy_emb, basis, imp_ham, \
#                lattice=lat, last_dmu=last_dmu, int_bath=True, \
#                                      solver=solver_info[0], solver_args=solver_info[1], labels=lo_labels)
#        energy_full *= lat.nscsites
#        return rho_full, energy_full
#
# Note here that the effect of environment included in this step is at the meanfield level. So if we
# stop the iteration here, the results will be that of the single-shot DMET.

# In the self-consistent DMET, the interaction between the environment and the impurity is improved
# iteratively. In this method, a correlation potential is introduced to account for the interactions
# between the impurity and its environment. We start with an initial guess of zero for this
# correlation potential and optimize it by minimizing the difference between density matrices
# obtained from the mean-field Hamiltonian and the impurity Hamiltonian. Let's initialize the
# correlation potential and define a function to optimize it.
#
# .. code-block:: python
#
#    def initialize_vcor(lat):
#        v_cor = dmet.VcorLocal(restricted=True, bogoliubov=False, nscsites=lat.nscsites)
#        v_cor.assign(np.zeros((2, lat.nscsites, lat.nscsites)))
#        return v_cor
#
#    def fit_correlation_potential(rho_emb, lat, basis, v_cor):
#        vcor_new, err = dmet.FitVcor(rho_emb, lat, basis, \
#                    v_cor, beta=np.inf, filling=filling, MaxIter1=300, MaxIter2=0)
#
#        dVcor_per_ele = np.max(np.abs(vcor_new.param - v_cor.param))
#        v_cor.update(vcor_new.param)
#        return v_cor, dVcor_per_ele
#
# Now, we have defined all the ingredients of DMET. We can set up the self-consistency loop to get
# the full execution. We set up this loop by defining the maximum number of iterations and a
# convergence criteria. We use both energy and correlation potential as our convergence parameters,
# so we define the initial values and convergence tolerance for both.
#
# .. code-block:: python
#
#    import libdmet.dmet.Hubbard as dmet
#
#    max_iter = 10 # maximum number of iterations
#    e_old = 0.0 # initial value of energy
#    v_cor = initialize_vcor(lat) # initial value of correlation potential
#    dVcor_per_ele = None # initial value of correlation potential per electron
#    vcor_tol = 1.0e-5 # tolerance for correlation potential convergence
#    energy_tol = 1.0e-5 # tolerance for energy convergence
#    mu = 0 # initial chemical potential
#    last_dmu = 0.0 # change in chemical potential
#
# Now we set up the iterations in a loop.
#
# .. code-block:: python
#
#    for i in range(max_iter):
#        rho, mu, scf_result, imp_ham, basis = construct_impurity_hamiltonian(lat,
#                                         v_cor, filling, mu, last_dmu) # construct impurity Hamiltonian
#        rho_emb, energy_emb, imp_ham, last_dmu, solver_info = solve_impurity_hamiltonian(lat, cell,
#                                         basis, imp_ham, last_dmu, scf_result) # solve impurity Hamiltonian
#        rho_full, energy_full = solve_full_system(lat, rho_emb, energy_emb, basis, imp_ham,
#                                         last_dmu, solver_info, lo_labels) # include the environment interactions
#        v_cor, dVcor_per_ele = fit_correlation_potential(rho_emb,
#                                         lat, basis, v_cor) # fit correlation potential
#
#        dE = energy_full - e_old
#        e_old = energy_full
#        if dVcor_per_ele < vcor_tol and abs(dE) < energy_tol:
#            print("DMET Converged")
#            print("DMET Energy per cell: ", energy_full)
#            break
#
# This concludes the DMET procedure.
#
# At this point, we should note that we are still limited by the number of orbitals we can have in
# the impurity because the cost of using a high-level solver such as FCI increases exponentially
# with increase in system size. One way to solve this problem could be through the use of
# quantum computing algorithm as solver.
#
# Next, we see how we can convert this impurity Hamiltonian to a qubit Hamiltonian through PennyLane
# to pave the path for using it with quantum algorithms. The hamiltonian object generated above
# provides us with one-body and two-body integrals along with the nuclear repulsion energy which can
# be accessed as follows:
#
# .. code-block:: python
#
#    from pyscf import ao2mo
#    norb = imp_ham.norb
#    h1 = imp_ham.H1["cd"]
#    h2 = imp_ham.H2["ccdd"][0]
#    h2 = ao2mo.restore(1, h2, norb) # Get the correct shape based on permutation symmetry
#
# These one-body and two-body integrals can then be used to generate the qubit Hamiltonian in
# PennyLane. We then diagonaliz it to get the eigenvalues and show that the lowest eigenvalue
# matches the energy we obtained for the embedded system above.
#
# .. code-block:: python
#
#    import pennylane as qml
#    from pennylane.qchem import one_particle, two_particle, observable
#
#    t = one_particle(h1[0])
#    v = two_particle(np.swapaxes(h2, 1, 3)) # Swap to physicist's notation
#    qubit_op = observable([t,v], mapping="jordan_wigner")
#    eigval_qubit = qml.eigvals(qml.SparseHamiltonian(qubit_op.sparse_matrix(), wires = qubit_op.wires))
#    print("eigenvalue from PennyLane: ", eigval_qubit)
#    print("embedding energy: ", energy_emb)
#
# We can also get ground state energy for the system from this value by solving for the full system
# as done above in the self-consistency loop using solve_full_system function.
#
# Conclusion
# ^^^^^^^^^^^^^^
# The density matrix embedding theory is a robust method designed to tackle simulation of complex
# systems by dividing them into subsystems. It is specifically suited for studying the ground state
# properties of a highly-correlated system. It provides for a computationally efficient alternative
# to dynamic quantum embedding schemes such as dynamic meanfield theory(DMFT), as it uses density
# matrix for embedding instead of the Green's function and has limited number of bath orbitals. It
# has been successfully used for studying various strongly correlated molecular and periodic systems.
#
# References
# ----------
#
# .. [#SWouters]
#    Sebastian Wouters, Carlos A. Jiménez-Hoyos, *et al.*, 
#    "A practical guide to density matrix embedding theory in quantum chemistry."
#    `ArXiv <https://arxiv.org/pdf/1603.08443>`__.
#     
#     
# .. [#pyscf]
#     Qiming Sun, Xing Zhang, *et al.*, "Recent developments in the PySCF program package."
#     `ArXiv <https://arxiv.org/pdf/2002.12531>`__.
#