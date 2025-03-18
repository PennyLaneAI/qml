r"""Density Matrix Embedding Theory (DMET)
=========================================
Materials simulation presents a crucial challenge in quantum chemistry, as understanding and predicting the properties of 
complex materials is essential for advancements in technology and science. While Density Functional Theory (DFT) is 
the current workhorse in this field due to its balance between accuracy and computational efficiency, it often falls short in 
accurately capturing the intricate electron correlation effects found in strongly correlated materials. As a result, 
researchers often turn to more sophisticated methods, such as full configuration interaction or coupled cluster theory, 
which provide better accuracy but come at a significantly higher computational cost. Embedding theories provide a balanced 
midpoint solution that enhances our ability to simulate materials accurately and efficiently. The core idea behind embedding 
is to treat the strongly correlated subsystem accurately using high-level quantum mechanical methods while approximating
the effects of the surrounding environment in a way that retains computational efficiency. 
Density matrix embedding theory(DMET) is one such efficient wave-function-based embedding approach to treat strongly 
correlated systems. Here, we present a demonstration of how to run DMET calculations through an existing library called 
libDMET, along with the intructions on how we can use the generated Hamiltonian with PennyLane to use it with quantum 
computing algorithms. We begin by providing a high-level introduction to DMET, followed by a tutorial on how to set up 
a DMET calculation.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# Theory
# ------
#
######################################################################
# Implementation
# --------------
# We now use what we have learned to set up a DMET calculation for $H_6$ system.
#
# Constructing the system
# ^^^^^^^^^^^^^^^^^^^^^^^
# We begin by defining a periodic system using the PySCF interface to create a cell object
# representing a hydrogen chain with 6 atoms. The lattice vectors are specified to define the
# geometry of the unit cell, within this unit cell, we place two hydrogen atoms: one at the origin
# (0.0, 0.0, 0.0) and the other at (0.0, 0.0, 0.75), corresponding to a bond length of
# 0.75 Å. We further specify a k-point mesh of [1, 1, 3], which represents the number of
# k-points sampled in each spatial direction for the periodic Brillouin zone. Finally, we
# construct a Lattice object from the libDMET library, associating it with the defined cell
# and k-mesh, which allows for the use of DMET in studying the properties of the hydrogen
# chain system.
import numpy as np
from pyscf.pbc import gto, df, scf, tools
from libdmet.system import lattice

cell = gto.Cell()
cell.a = ''' 10.0    0.0     0.0                                                                                                                                                                                      
             0.0     10.0    0.0                                                                                                                                                                                      
             0.0     0.0     1.5 '''
cell.atom = ''' H 0.0      0.0      0.0                                                                                                                                                                               
                H 0.0      0.0      0.75 '''
cell.basis = '321g'
cell.build(unit='Angstrom')

kmesh = [1, 1, 3]
Lat = lattice.Lattice(cell, kmesh)
filling = cell.nelectron / (Lat.nscsites*2.0)
kpts = Lat.kpts
#
######################################################################
# Now we have a description of our system and can start obtaining the fragment and bath orbitals.
#
# Constructing the bath orbitals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We perform a mean-field calculation on the whole system through Hartree-Fock with density
# fitted integrals using PySCF.
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = 'gdf_ints.h5'
gdf.build()

kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
kmf.with_df = gdf
kmf.with_df._cderi = 'gdf_ints.h5'
kmf.conv_tol = 1e-12
kmf.max_cycle = 200
kmf.kernel()

######################################################################
# In quantum chemistry calculations, we can choose the bath and impurity by looking at the
# labels of orbitals. With this code, we can get the orbitals and separate the valence and
# virtual labels for each atom in the unit cell as shown below. This information helps us
# identify the orbitals to be included in the impurity, bath and unentangled environment.
# In this example, we choose to keep all the valence orbitals in the unit cell in the
# impurity, while the bath contains the virtual orbitals, and the orbitals belonging to the
# rest of the supercell become part of the unentangled environment.
from libdmet.lo.iao import reference_mol, get_labels, get_idx

labels, val_labels, virt_labels = get_labels(cell, minao="MINAO")

ncore = 0
nval  = len(val_labels)
nvirt = len(virt_labels)

Lat.set_val_virt_core(nval, nvirt, ncore)
print(labels, nval, nvirt)
######################################################################
# Further, we rotate the integrals into the embedding basis and obtain the rotated Hamiltonian

from libdmet.basis_transform import make_basis

C_ao_iao, C_ao_iao_val, C_ao_iao_virt, lo_labels = make_basis.get_C_ao_lo_iao(Lat, kmf, minao="MINAO", full_return=True, return_labels=True)
C_ao_lo = Lat.symmetrize_lo(C_ao_iao)
Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

######################################################################
# DMET
# ^^^^^^^^^^^^^^^^^^^^
# Now that we have a description of our fragment and bath orbitals, we can implement DMET. 
# We implement each step of the process in a function and 
# then call these functions to perform the calculations. This can be done once for one iteration,
# referred to as single-shot DMET or we can call them iteratively to perform self-consistent DMET.
# Let's start by constructing the impurity Hamiltonian, 
def construct_impurity_hamiltonian(Lat, vcor, filling, mu, last_dmu, int_bath=True):

    rho, mu, res = dmet.HartreeFock(Lat, vcor, filling, mu,
                                    ires=True, labels=lo_labels)
    
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, int_bath=int_bath)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)

    return rho, mu, res, ImpHam, basis

# Next, we solve this impurity Hamiltonian with a high-level method, the following function defines
# the electronic structure solver for the impurity, provides an initial point for the calculation and 
# passes the Lattice information to the solver.
def solve_impurity_hamiltonian(Lat, cell, basis, ImpHam, last_dmu, res):

    solver = dmet.impurity_solver.FCI(restricted=True, tol=1e-13)
    basis_k = Lat.R2k_basis(basis)

    solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, Lat.nkpts*cell.nelectron), \
               "dm0": dmet.foldRho_k(res["rho_k"], basis_k)}
    
    rhoEmb, EnergyEmb, ImpHam, dmu = dmet.SolveImpHam_with_fitting(Lat, filling, 
        ImpHam, basis, solver, solver_args=solver_args)

    last_dmu += dmu
    return rhoEmb, EnergyEmb, ImpHam, last_dmu, [solver, solver_args]

# We can now calculate the properties for our embedded system through this embedding density matrix. Final step
# in single-shot DMET is to include the effect of environment in the final expectation value, so we define a 
# function for the same which returns the density matrix and energy for the whole/embedded system
def solve_full_system(Lat, rhoEmb, EnergyEmb, basis, ImpHam, last_dmu, solver_info, lo_labels):
    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, \
            lattice=Lat, last_dmu=last_dmu, int_bath=True, \
                                  solver=solver_info[0], solver_args=solver_info[1], labels=lo_labels)
    return rhoImp, EnergyImp
    
# We must note here that the effect of environment included in the previous step is
# at the meanfield level. We can look at a more advanced version of DMET and improve this interaction
#  with the use of self-consistency, referred to
# as self-consistent DMET, where a correlation potential is introduced to account for the interactions 
# between the impurity and its environment. We start with an initial guess of zero for this correlation
#  potential and optimize it by minimizing the difference between density matrices obtained from the
#  mean-field Hamiltonian and the impurity Hamiltonian. Now, we initialize the correlation potential
# and define a function to optimize it.
import libdmet.dmet.Hubbard as dmet
vcor = dmet.VcorLocal(restricted=True, bogoliubov=False, nscsites=Lat.nscsites)
z_mat = np.zeros((2, Lat.nscsites, Lat.nscsites))
vcor.assign(z_mat)
def fit_correlation_potential(rhoEmb, Lat, basis, vcor):
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta=np.inf, filling=filling, MaxIter1=300, MaxIter2=0)

    dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))
    vcor.update(vcor_new.param)
    return vcor, dVcor_per_ele

# Now, we have defined all the ingredients of DMET, we can set up the self-consistency loop to get
# the full execution. We set up this loop by defining the maximum number of iterations and a convergence
# criteria. Here, we are using both energy and correlation potential as our convergence parameters, so we 
# define the initial values and convergence tolerance for both.

maxIter = 10
E_old = 0.0
dVcor_per_ele = None
u_tol = 1.0e-5
E_tol = 1.0e-5
mu = 0
last_dmu = 0.0
for i in range(maxIter):
    rho, mu, res, ImpHam, basis = construct_impurity_hamiltonian(Lat, vcor, filling, mu, last_dmu)
    rhoEmb, EnergyEmb, ImpHam, last_dmu, solver_info = solve_impurity_hamiltonian(Lat, cell, basis, ImpHam, last_dmu, res)
    rhoImp, EnergyImp = solve_full_system(Lat, rhoEmb, EnergyEmb, basis, ImpHam, last_dmu, solver_info, lo_labels)
    vcor, dVcor_per_ele = fit_correlation_potential(rhoEmb, Lat, basis, vcor)

    dE = EnergyImp - E_old
    E_old = EnergyImp
    if dVcor_per_ele < u_tol and abs(dE) < E_tol:
        print("DMET Converged")
        print("DMET Energy per cell: ", EnergyImp*Lat.nscsites/1)
        break



