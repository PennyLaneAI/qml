r"""Quantum Defect Embedding Theory (QDET)
=========================================
Efficient simulation of complex quantum systems remains a significant challenge in chemistry and
physics. These simulations often require computationally intractable methods for a complete
solution. However, many interesting problems in quantum chemistry and condensed matter physics
feature a strongly correlated region, which requires accurate quantum treatment, embedded within a
larger environment that could be properly treated with cheaper approximations.  Examples of such
systems include point defects in materials [#Galli]_, active site of catalysts [#SJRLee]_, surface phenomenon such
as adsorption [#Gagliardi]_ and many more. Embedding theories serve as powerful tools for effectively
addressing such problems.

The core idea behind embedding methods is to partition the system and treat the strongly correlated
subsystem accurately, using high-level quantum mechanical methods, while approximating the effects
of the surrounding environment in a way that retains computational efficiency. In this demo, we show
how to implement the quantum defect embedding theory (QDET). The method has been successfully
applied to study defects in CaO and to calculate excitations of the negatively charged NV center in diamond. An important advantage of QDET is its compatibility with quantum
algorithms as we explain in the following sections. The method can be implemented for calculating
a variety of ground state, excited state and dynamic properties of materials. These make QDET a
powerful method for affordable quantum simulation of materials.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# Theory
# ------
# The core idea in QDET is to construct an effective Hamiltonian that describes the impurity
# subsystem and also accounts for the interactions between the impurity subsystem and the
# environment as
#
# .. math::
#
#     H^{eff} = \sum_{ij}^{A} t_{ij}^{eff}a_i^{\dagger}a_j + \frac{1}{2}\sum_{ijkl}^{A} v_{ijkl}^{eff}a_i^{\dagger}a_{j}^{\dagger}a_ka_l,
#
# where :math:`t_{ij}^{eff}` and :math:`v_{ijkl}^{eff}` represent the effective one-body and
# two-body integrals, respectively, and :math:`ijkl` span over the orbitals inside the impurity.
# This Hamiltonian describes a simplified representations of the complex quantum systems that is
# computationally tractable and properly captures the essential physics of the problem. The
# effective integrals :math:`t, v` can be obtained by fitting experimental results or may be
# derived from first-principles calculations [].
#
# A QDET calculation typically starts by obtaining a meanfield approximation of the whole system
# using efficient quantum chemistry methods such as density functional theory. These calculations
# provide a set of orbitals which can be split into impurity and bath orbitals. These orbitals are
# used to construct the effective Hamiltonian which is finally solved by using either a high level
# classical method or a quantum algorithm. Let's implement these steps for an example!
#
# Implementation
# --------------
# We implement QDET to compute the excitation energies of a negatively charged nitrogen-vacancy
# defect in diamond [].
#
# Mean field calculations
# ^^^^^^^^^^^^^^^^^^^^^^^
# We use density functional theory To obtain a mean-field description of the whole system. The DFT
# calculations are performed with the QUANTUM ESPRESSO package. This requires downloading
# parameters needed for each atom type in the system from the QUANTUM ESPRESSO
# `database <https://www.quantum-espresso.org/pseudopotentials/>`_. We have carbon and nitrogen in
# our system which can be downloaded with

wget -N -q http://www.quantum-simulation.org/potentials/sg15_oncv/upf/C_ONCV_PBE-1.2.upf
wget -N -q http://www.quantum-simulation.org/potentials/sg15_oncv/upf/N_ONCV_PBE-1.2.upf

##############################################################################
# Next, we need to create the input file for running QUANTUM ESPRESSO. The input file ``pw.in``
# contains information about the system and details of the DFT calculations. More details on
# how to construct the input file can be found in QUANTUM ESPRESSO
# `documentation <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`_ page.
#
# We can now perform the DFT calculations by running the executable code ``pw.x`` on the input file:
mpirun -n 2 pw.x -i pw.in > pw.out


# Identify the impurity
# ^^^^^^^^^^^^^^^^^^^^^
# Once we have obtained the meanfield description, we can identify our impurity by finding
# the states that are localized in real space. We can identify these localized states using the
# localization factor defined as []:
#
# .. math::
#
#     L_n = \int_{V \in \ohm} d^3 r |\Psi_n^{KS}(r)|^
#
# We will use the WEST program to compute the localization factor. This requires creating another
# input file ``westpp.in`` as shown below.

westpp_control:
  westpp_calculation: L # triggers the calculation of the localization factor
  westpp_range:         # defines the range of states toe compute the localization factor
  - 1                   # start from the first state
  - 176                 # use all the 176 state
  westpp_box:           # specifies the parameter of the box in atomic units for integration
  - 6.19                #
  - 10.19
  - 6.28
  - 10.28
  - 6.28
  - 10.28

##############################################################################
# We can execute this calculation as

mpirun -n 2 westpp.x -i westpp.in > westpp.out

##############################################################################
# This creates a file named ``west.westpp.save/westpp.json``. Since computational resources required
# to run the calculation are large, the WEST output file needed for the next step can be
# directly downloaded as:

mkdir -p west.westpp.save
wget -N -q https://west-code.org/doc/training/nv_diamond_63/box_westpp.json -O west.westpp.save/westpp.json

##############################################################################
# We can now plot the computed localization factor for each of the states:

import json
import numpy as np
import matplotlib.pyplot as plt

with open('west.westpp.save/westpp.json','r') as f:
    data = json.load(f)

y = np.array(data['output']['L']['K000001']['local_factor'],dtype='f8')
x = np.array([i+1 for i in range(y.shape[0])])

plt.plot(x,y,'o')
plt.axhline(y=0.08,linestyle='--',color='red')

plt.xlabel('KS index')
plt.ylabel('Localization factor')

plt.show()

##############################################################################
# From this plot, it is easy to see that Kohn-Sham orbitals can be catergorized as orbitals
# with low and high localization factor. For the purpose of defining an impurity, we need
# highly localized orbitals, so for this we set a cutoff of 0.06 and choose the orbitals
# that have a localization factor > 0.06 for our active space. We'll use these orbitals for
# the calculation of the parameters for the effective Hamiltonian in the following section.
#
# Effective Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^
# The next and probably most important steps in QDET is to define the effective one-body and
# two-body integrals for the impurity. The effective two-body integrals, $v^{eff}$ are computed
# first as matrix elements of the partially screened static Coulomb potential $W_0^{R}$.
# $$v_{ijkl}^{eff} = [W_0^{R}]_{ijkl}$$
# $W_0^R$, results from screening the bare Coulomb potential, $v$, with the reduced polarizability,
# $P_0^R = P - P_{imp}$, where $P$ is the system's polarizability and $P_{imp}$ is the impurity's
# polarizability. However, this definition of the effective interaction, $v_{eff}$, introduces
# double counting of electrostatic and exchange-correlation effects for the impurity: once via
# density functional theory (DFT) and again via the high-level method.
# Therefore, once $v^{eff}$ is obtained, the one-body term $t^{eff}$ is then modified by subtracting
# from the Kohn-Sham Hamiltonian the term accounting for electrostatic and exchange
# correlation interactions in the active space.
#
# $$t_{ij}^{eff} = H_{ij}^{KS} - t_{ij}^{dc}$$
#
# In WEST, these parameters for the effective Hamiltonian are calculated by using the wfreq.x
# executable. The program will: (i) compute the quasiparticle energies, (ii) compute the
# partially screened Coulomb potential, and (iii) finally compute the parameters of the
# effective Hamiltonian. The input file for such a calculation is shown below:

wstat_control:
  wstat_calculation: S
  n_pdep_eigen: 512
  trev_pdep: 0.00001

wfreq_control:
  wfreq_calculation: XWGQH
  macropol_calculation: C
  l_enable_off_diagonal: true
  n_pdep_eigen_to_use: 512
  qp_bands: [87, 122, 123, 126, 127, 128]
  n_refreq: 300
  ecut_refreq: 2.0

##############################################################################
# We now construct the effective Hamiltonian:

from westpy.qdet import QDETResult

mkdir -p west.wfreq.save
wget -N -q https://west-code.org/doc/training/nv_diamond_63/wfreq.json -O west.wfreq.save/wfreq.json

effective_hamiltonian = QDETResult(filename='west.wfreq.save/wfreq.json')

##############################################################################
# The final step is to solve for this effective Hamiltonian using a high level method. We can
# use the WESTpy package as:

solution = effective_hamiltonian.solve()

##############################################################################
# We can also use this effective Hamiltonian with a quantum algorithm through PennyLane. This
# requires representing it in the qubit Hamiltonian format for PennyLane and can be done as follows:

from pennylane.qchem import one_particle, two_particle, observable
import numpy as np

effective_hamiltonian = QDETResult(filename='west.wfreq.save/wfreq.json')

one_e, two_e = effective_hamiltonian.h1e, effective_hamiltonian.eri

t = one_particle(one_e[0])
v = two_particle(np.swapaxes(two_e[0][0], 1, 3))
qubit_op = observable([t, v], mapping="jordan_wigner")

##############################################################################
# The ground state energy of the Hamiltonian is identical to the one obtained before.
#
# Conclusion
# ----------
# The quantum density embedding theory is a novel framework for simulating strongly correlated
# quantum systems and has been successfully used for studying defects in solids. Applicability of
# QDET however is not limited to defects, it can be used for other systems where a strongly correlated subsystem
#  is embedded in a weakly correlated environment. QDET is able to surpass the problem of correction of double
# counting of interactions within the active space faced by some other embedding theories 
# such as DFT+DMFT.  Green's function based formulation of QDET ensures exact removal of double counting 
# corrections at GW level of theory, thus removing the approximation present in the initial DFT based formulation. This 
# formulation also helps capture the response properties and provides access to excited state properties.
# Another major advantage of QDET is the ease with which it can be used with quantum computers in a hybrid framework. 
# Therefore, We can conlcude here that QDET is a powerful embedding approach for simulating complex quantum systems.
#  
# References
# ----------
#
# .. [#ashcroft]
#
#     N. W. Ashcroft, D. N. Mermin,
#     "Solid State Physics", Chapter 4, New York: Saunders College Publishing, 1976.
#
# .. [#Galli]
#
#    Joel Davidsson, Mykyta Onizhuk, *et al.*, "Discovery of atomic clock-like spin defects in simple oxides from first principles"
#    `ArXiv <https://arxiv.org/pdf/2302.07523>`__.
#
# .. [#SJRLee]
#    Sebastian J. R. Lee, Feizhi Ding, *et al.*, "Analytical Gradients for Projection-Based Wavefunction-in-DFT Embedding."
#    `ArXiv <https://arxiv.org/pdf/1903.05830>`__.
#
# .. [#Gagliardi]
#    Abhishek Mitra, Matthew Hermes, *et al.*, "Periodic Density Matrix Embedding for CO Adsorption on the MgO(001)Surface."
#    `ChemRxiv <https://chemrxiv.org/engage/chemrxiv/article-details/62b0b0c40bba5d82606d2cae>`__.
#
# About the authors
# -----------------
#
