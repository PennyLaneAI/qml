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
# rest of the supercell becomes part of the unentangled environment.
from libdmet.lo.iao import reference_mol, get_labels, get_idx

# aoind = cell.aoslice_by_atom()
# ao_labels = cell.ao_labels()

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
# Self-consistent DMET
# ^^^^^^^^^^^^^^^^^^^^
# Now that we have a description of our fragment and bath orbitals, we can implement DMET
# self-consistently. We implement each step of the process in a function and then iteratively
# call this functions in a loop to perform the calculations.
#
def mean_field():
    rho, Mu, res = dmet.HartreeFock(Lat, vcor, filling, mu,
                                    beta=beta, ires=True, labels=lo_labels)

    return rho, Mu, res