r"""Density Matrix Embedding Theory (DMET)
=========================================
Materials simulation presents a crucial challenge in quantum chemistry. Density Functional Theory
(DFT) is currently the workhorse for simulating materials due to its balance between accuracy and
computational efficiency. However, it often falls short in accurately capturing the intricate
electron correlation effects found in strongly correlated materials because of its meanfield nature.
As a result, researchers often turn to wavefunction-based methods, such as configuration interaction
or coupled cluster, which provide better accuracy but come at a
significantly higher computational cost.

Embedding theories provide a balanced midpoint solution that enhances our ability to simulate
materials accurately and efficiently. The core idea behind embedding is that the system is divided
into two parts: an impurity which is a strongly correlated subsystem that requires a high accuracy
description and an environment which can be treated with more approximate but computationally efficient level of theory.

Embedding approaches differ in how they capture the environment's effect on the embedded region,
and can be broadly categorized as density in density, wavefunction in wavefunction and Green's function based embeddings.
In this demo, we will focus on Density matrix embedding theory (DMET) [#SWouters]_, a
wavefunction based approach embedding the ground state density matrix.
We provide a brief introduction of the method and demonstrate how to run
DMET calculations to construct a Hamiltonian that can be used for quantum algorithms.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

#############################################
# Theory
# ------
# The wave function for the embedded system composed of the impurity and the environment can be
# simply represented as
#
# .. math::
#
#      | \Psi \rangle = \sum_i^{N_I} \sum_j^{N_E} \psi_{ij} | I_i \rangle | E_j \rangle,
#
# where :math:`I_i` and :math:`E_j` are basis states of the impurity :math:`I` and environment
# :math:`E`, respectively, :math:`\psi_{ij}` is the matrix of coefficients and :math:`N` is the
# number of sites, e.g., orbitals. The key idea in DMET is to perform a singular value decomposition
# of the coefficient matrix :math:`\psi_{ij} = \sum_{\alpha} U_{i \alpha} \lambda_{\alpha} V_{\alpha j}`
# and rearrange the wave functions such that
#
# .. math::
#
#      | \Psi \rangle = \sum_{\alpha}^{N} \lambda_{\alpha} | A_{\alpha} \rangle | B_{\alpha} \rangle,
#
# where :math:`A_{\alpha} = \sum_i U_{i \alpha} | I_i \rangle` are states obtained from rotations of
# :math:`I_i` to a new basis, and :math:`B_{\alpha} = \sum_j V_{j \alpha} | E_j \rangle` are bath
# states representing the portion of the environment that interacts with the impurity. Note that the
# number of bath states is equal to the number of fragment states, regardless of the size of the
# environment. This new decomposition is the Schmidt decomposition of the system wave function.
#
# We are now able to project the full Hamiltonian to the space of impurity and bath states, known as
# embedding space.
#
# .. math::
#
#      \hat{H}_{emb} = \hat{P}^{\dagger} \hat{H} \hat{P}
#
# where :math:`P = \sum_{\alpha \beta} | A_{\alpha} B_{\beta} \rangle \langle A_{\alpha} B_{\beta}|`
# is a projection operator. A key point about this representation is that the wave function,
# :math:`|\Psi \rangle`, is the ground state of both the full system Hamiltonian :math:`\hat{H}` and
# the smaller embedded Hamiltonian :math:`\hat{H}_{emb}` [arXiv:2108.08611].
#
# Note that the Schmidt decomposition requires apriori knowledge of the wavefunction. To alleviate
# this, DMET operates through a systematic iterative approach, starting with a meanfield description
# of the wavefunction and refining it through feedback from solution of the impurity Hamiltonian.
#
# Implementation
# --------------
# The DMET procedure starts by getting an approximate description of the system's wavefunction.
# Subsequently, this approximate wavefunction is partitioned with Schmidt decomposition to get
# the impurity and bath orbitals. These orbitals are then employed to define an approximate projector
# :math:`P`, which in turn is used to construct the embedded Hamiltonian. High accuracy methods such as
# post-Hartree-Fock methods, exact diagonalisation, or accurate quantum algorithms are employed to find
# the energy spectrum of the embedded Hamiltonian. The results are used to
# re-construct the projector and the process is repeated until the wave function converges. Let's now
# take a look at the implementation of these steps for a toy system with 8 Hydrogen atoms.
# We use the programs PySCF [#pyscf]_ and libDMET [#libdmet]_ [#libdmet2]_ which can be installed with
#
# .. code-block:: python
#
# pip install pyscf
# pip install git+https://github.com/gkclab/libdmet_preview.git@main
#
# Constructing the system
# ^^^^^^^^^^^^^^^^^^^^^^^
# We begin by defining a hydrogen chain with 8 atoms using PySCF. This is done by creating
# a ``Cell`` object containing 8 Hydrogen atoms. The system is arranged with a central H$_4$
# chain featuring a uniform 0.75 Å H-H bond length, flanked by two H$_2$ molecules at its termini.
# Then, we construct a ``Lattice`` object from the libDMET library, associating it with
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
#                  0.0     10.0    0.0
#                  0.0     0.0     12.0 ''' # lattice vectors for unit cell
#
#    cell.atom = ''' H 0.0      0.0      0.0
#                    H 0.0      0.0      0.75
#                    H 0.0      0.0      3.0
#                    H 0.0      0.0      3.75
#                    H 0.0      0.0      4.5
#                    H 0.0      0.0      5.25
#                    H 0.0      0.0      7.5
#                    H 0.0      0.0      8.25''' # coordinates of atoms in unit cell
#
#    cell.basis = '321g'
#    cell.build(unit='Angstrom')
#
#    kmesh = [1, 1, 1] # number of k-points in xyz direction
#
#    lattice = lattice.Lattice(cell, kmesh)
#    filling = cell.nelectron / (lattice.nscsites * 2.0)
#    kpts = lattice.kpts
#
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
#    c_ao_iao, _, _ = make_basis.get_C_ao_lo_iao(
#    lattice, kmf, minao="MINAO", full_return=True)
#    c_ao_lo = lattice.symmetrize_lo(c_ao_iao)
#    lattice.set_Ham(kmf, gdf, c_ao_lo, eri_symmetry=4) # rotate integral tensors to IAO basis
#
# Next, we identify the orbital labels for each atom in the unit cell and define the impurity and bath
# by looking at the labels. In this example, we choose to keep the :math:`1s` orbitals in the central
# :math:`H_4` chain in the impurity, while the bath contains the :math:`2s` orbitals, and the orbitals belonging
# to the terminal Hydrogen molecules form the unentangled environment. The indices for the valence and virtual orbitals
# corresponding to the impurity and the bath, respectively, can be obtained using
# the get_idx function.
#
# .. code-block:: python
#
#    from libdmet.lo.iao import get_labels, get_idx
#
#    aoind = cell.aoslice_by_atom()
#    labels, val_labels, virt_labels = get_labels(cell, minao="MINAO")
#    imp_idx = get_idx(val_labels, atom_num=[2,3,4,5])
#    bath_idx = get_idx(virt_labels, atom_num=[2,3,4,5], offset=len(val_labels))
#    ncore = []
#    lattice.set_val_virt_core(imp_idx, bath_idx, ncore)
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
#    def construct_impurity_hamiltonian(lattice, v_cor, filling, mu, last_dmu, int_bath=True):
#        r"""A function to construct the impurity Hamiltonian
#
#        Args:
#            lattice: Lattice object containing information about the system
#            v_core: correlation potential
#            filling: average number of electrons occupying the orbitals
#            mu: chemical potential
#            last_dmu: change in chemical potential from last iterations
#            int_bath: Flag to determine whether we are using an interactive bath
#
#        Returns:
#            rho: mean-field density matrix
#            mu: chemical potential
#            scf_result: object containing the results of meanfield calculation
#            imp_ham: impurity Hamiltonian
#            basis: rotation matrix for embedding basis
#        """
#
#        rho, mu, scf_result = dmet.HartreeFock(lattice, v_cor, filling, mu,
#                                        ires=True)
#        imp_ham, _, basis = dmet.ConstructImpHam(lattice, rho, v_cor, int_bath=int_bath)
#        imp_ham = dmet.apply_dmu(lattice, imp_ham, basis, last_dmu)
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
#    def solve_impurity_hamiltonian(lattice, cell, basis, imp_ham, filling_imp, last_dmu, scf_result):
#        r"""A function to solve impurity Hamiltonian
#
#        Args:
#            lattice: Lattice object containing information about the system
#            cell: Cell object containing information about the unit cell
#            basis: rotation matrix for embedding basis
#            imp_ham: impurity Hamiltonian
#            last_dmu: change in chemical potential from previous iterations
#            scf_result: object containing the results of meanfield calculation
#
#        Returns:
#            rho_emb: density matrix for embedded system
#            energy_emb: energy for embedded system
#            imp_ham: impurity Hamiltonian
#            last_dmu: change in chemical potential from last iterations
#            solver_info: a list containing information about the solver
#        """      
#
#        solver = dmet.impurity_solver.FCI(restricted=True, tol=1e-8) # impurity solver
#        basis_k = lattice.R2k_basis(basis) # basis in k-space
#
#        solver_args = {"nelec": min((lattice.ncore+lattice.nval)*2, lattice.nkpts*cell.nelectron), \
#                   "dm0": dmet.foldRho_k(scf_result["rho_k"], basis_k)}
#
#        rho_emb, energy_emb, imp_ham, dmu = dmet.SolveImpHam_with_fitting(lattice, filling_imp,
#            imp_ham, basis, solver, solver_args=solver_args)
#
#        last_dmu += dmu
#        solver_info = [solver, solver_args]
#        return rho_emb, energy_emb, imp_ham, last_dmu, solver_info
#
# The final step is to include the effect of the environment in the expectation value. So we define
# a function which returns the density matrix and energy for the whole system.
#
# .. code-block:: python
#
#    def solve_full_system(lattice, rho_emb, energy_emb, basis, imp_ham, last_dmu, solver_info):
#        r"""A function to solve impurity Hamiltonian
#
#        Args:
#            lattice: Lattice object containing information about the system
#            rho_emb: density matrix for embedded system
#            energy_emb: energy for embedded system
#            basis: rotation matrix for embedding basis
#            imp_ham: impurity Hamiltonian
#            last_dmu: change in chemical potential from last iterations
#            solver_info: a list containing information about the solver
#
#        Returns:
#            energy: Ground state energy of the system
#        """
#
#        from libdmet.routine.slater import get_E_dmet_HF
#
#        rho_imp, energy_imp, nelec_imp = \
#                dmet.transformResults(rho_emb, energy_emb, basis, imp_ham, \
#                lattice=lattice, last_dmu=last_dmu, int_bath=True, \
#                             solver=solver_info[0], solver_args=solver_info[1])
#        energy_imp_fci = energy_imp * lattice.nscsites # Energy of impurity at FCI level
#        energy_imp_scf = get_E_dmet_HF(basis, lattice, imp_ham, last_dmu, solver_info[0])
#        energy = kmf.e_tot - kmf.energy_nuc() - energy_imp_scf + energy_imp_fci
#
#        return energy_imp_fci, energy
#
# Note here that the effect of environment included in this step is at the mean field level. So if we
# stop the iteration here, the results will be that of the single-shot DMET.
#
# In the self-consistent DMET, the interaction between the environment and the impurity is improved
# iteratively. In this method, a correlation potential is introduced to account for the interactions
# between the impurity and its environment, which can be represented as
#
# .. math::
#
#      \hat{C}_x = \sum_{kl} u_{kl}^{x}a_k^{\dagger}a_l
#
# We start with an initial guess of zero for the :math:`u` here,
# and optimize it by minimizing the difference between density matrices
# obtained from the mean-field Hamiltonian and the impurity Hamiltonian. Let's initialize the
# correlation potential and define a function to optimize it.
#
# .. code-block:: python
#
#    def initialize_vcor(lattice):
#        r"""A function to initialize the correlation potential
#
#        Args:
#            lattice: Lattice object containing information about the system
#
#        Returns:
#            Initial correlation potential
#        """
#
#        v_cor = dmet.VcorLocal(restricted=True, bogoliubov=False, nscsites=lattice.nscsites, idx_range=lattice.imp_idx)
#        v_cor.assign(np.zeros((2, lattice.nscsites, lattice.nscsites)))
#        return v_cor
#
#    def fit_correlation_potential(rho_emb, lattice, basis, v_cor):
#        r"""A function to solve impurity Hamiltonian
#
#        Args:
#            rho_emb: density matrix for embedded system
#            lattice: Lattice object containing information about the system
#            basis: rotation matrix for embedding basis
#            v_cor: correlation potential
#
#        Returns:
#            v_cor: correlation potential
#            dVcor_per_ele: change in correlation potential per electron
#        """
#
#        vcor_new, err = dmet.FitVcor(rho_emb, lattice, basis, \
#                    v_cor, beta=np.inf, filling=filling, MaxIter1=300, MaxIter2=0)
#
#        dVcor_per_ele = np.max(np.abs(vcor_new.param - v_cor.param))
#        v_cor.update(vcor_new.param)
#        return v_cor, dVcor_per_ele
#
# Now, we have defined all the ingredients of DMET. We can set up the self-consistency loop to get
# the full execution. We should note that dividing the system into multiple fragments might lead to wrong
# number of electrons, we keep this in check through the use of chemical potential, :math:`\mu`.
# We set up this loop by defining the maximum number of iterations and a
# convergence criteria. We use both energy and correlation potential as our convergence parameters,
# so we define the initial values and convergence tolerance for both.
#
# .. code-block:: python
#
#    import libdmet.dmet.Hubbard as dmet
#
#    max_iter = 10 # maximum number of iterations
#    e_old = 0.0 # initial value of energy
#    v_cor = initialize_vcor(lattice) # initial value of correlation potential
#    dVcor_per_ele = None # initial value of correlation potential per electron
#    vcor_tol = 1.0e-5 # tolerance for correlation potential convergence
#    energy_tol = 1.0e-5 # tolerance for energy convergence
#    mu = 0 # initial chemical potential
#    last_dmu = 0.0 # change in chemical potential
#
# Now we set up the iterations in a loop. We must note that defining an impurity which is a fragment of the unit cell,
# necessitates readjusting of the filling value for solution of impurity Hamiltonian,
# This filling value represents the average electron occupation, which scales proportional to the fraction
# of the unit cell included, while taking into account the different electronic species.
# For example, we are using half the number of Hydrogens inside the impurity,
# therefore, the filling becomes half as well.
#
# .. code-block:: python
#
#    for i in range(max_iter):
#        rho, mu, scf_result, imp_ham, basis = construct_impurity_hamiltonian(lattice,
#                           v_cor, filling, mu, last_dmu) # construct impurity Hamiltonian
#        filling_imp = filling * 0.5
#        rho_emb, energy_emb, imp_ham, last_dmu, solver_info = solve_impurity_hamiltonian(lattice, cell,
#                           basis, imp_ham, filling_imp, last_dmu, scf_result) # solve impurity Hamiltonian
#        energy_imp, energy = solve_full_system(lattice, rho_emb, energy_emb, basis, imp_ham,
#                           last_dmu, solver_info) # include the environment interactions
#        v_cor, dVcor_per_ele = fit_correlation_potential(rho_emb,
#                                         lattice, basis, v_cor) # fit correlation potential
#
#        dE = energy_imp - e_old
#        e_old = energy_imp
#        if dVcor_per_ele < vcor_tol and abs(dE) < energy_tol:
#            print("DMET Converged")
#            print("DMET Energy per cell: ", energy)
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
# PennyLane. We further diagonalize it to get the eigenvalues and show that the lowest eigenvalue
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
# as done above in the self-consistency loop using solve_full_system function. This qubit Hamiltonian
# is particularly relevant for hybrid flavor of DMET approach, where classical mean-field calculations
# are coupled with a quantum algorithm for the self-consistent solver. However, directly employing a
# quantum solver iteratively might not be the most efficient use of quantum resources. An alternative
# strategy involves using a classical solver for the iterative self-consistency between the impurity and
# the environment. The resulting converged impurity Hamiltonian can then be treated with a quantum algorithm
# for a more accurate final energy calculation."
#
# Conclusion
# ^^^^^^^^^^^^^^
# The density matrix embedding theory is a robust method designed to tackle simulation of complex
# systems by partitioning them into manageable subsystems. It is particularly well suited for studying
# the ground state properties of strongly correlated materials and molecules. DMET offers a compelling
# alternative to dynamic quantum embedding schemes such as dynamic meanfield theory(DMFT). By employing density
# matrix for embedding instead of the Green's function and utilizing Schmidt decomposition to get limited number
# of bath orbitals, DMET achieves significant computational advantage. Furthermore, a major strength lies
# in its compatibility with quantum algorithms, enabling the accurate study of smaller, strongly correlated
# subsystems on quantum computers while the environment is studied classically.
# Its successful application to a wide range of strongly correlated molecular and periodic systems underscores its
# power and versatility in electronic structure theory, paving the way for hybrid quantum-classical simulations of
# challenging materials.[#DMETQC]_

#
# References
# ----------
#
# .. [#SWouters]
#    Sebastian Wouters, Carlos A. Jiménez-Hoyos, *et al.*, 
#    "A practical guide to density matrix embedding theory in quantum chemistry."
#    `ArXiv <https://arxiv.org/pdf/1603.08443>`__.
#     
# .. [#pyscf]
#     Qiming Sun, Xing Zhang, *et al.*, "Recent developments in the PySCF program package."
#     `ArXiv <https://arxiv.org/pdf/2002.12531>`__.
#
# .. [#libdmet]
#     Zhi-Hao Cui, Tianyu Zhu, *et al.*, "Efficient Implementation of Ab Initio Quantum Embedding in Periodic Systems: Density Matrix Embedding Theory."
#     `ArXiv <https://arxiv.org/pdf/1909.08596>`__.
#
# .. [#libdmet2]
#    Tianyu Zhu, Zhi-Hao Cui, *et al.*, "Efficient Formulation of Ab Initio Quantum Embedding in Periodic Systems: Dynamical Mean-Field Theory."
#    `ArXiv <https://arxiv.org/pdf/1909.08592>`__.
#
#
# .. [#DMETQC]
#    Changsu Cao,Jinzhao Sun, *et al.*, "Ab initio simulation of complex solids on a quantum computer with orbital-based multifragment density matrix embedding"
#    `ArXiv <https://arxiv.org/pdf/2209.03202>`__.
#
