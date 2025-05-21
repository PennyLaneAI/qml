r"""Quantum Defect Embedding Theory (QDET)
=========================================
Performing efficient simulations of advanced materials and molecules remains a significant
challenge in quantum chemistry and condensed matter physics due to the prohibitive costs of
available methods. However, many interesting problems in quantum chemistry and condensed matter physics
feature a strongly correlated region, which requires accurate quantum treatment, embedded within a
larger environment that could be properly treated with cheaper approximations.  For example,
this is the case for point defects in materials [#Galli]_, active site of catalysts [#SJRLee]_, surface phenomenon such
as adsorption [#Gagliardi]_ and many more. Embedding theories serve as powerful tools for effectively
addressing such problems by capturing the strong electronic correlations in the active region with high accuracy
while accounting for the environment in a more approximate manner.

The core idea behind embedding methods is to partition the system and treat the strongly correlated
subsystem accurately, using high-level quantum mechanical methods, while approximating the effects
of the environment in a way that retains computational efficiency. In this demo, we show
how to implement the quantum defect embedding theory (QDET). This method has been successfully
applied to study defects in CaO [#Galli]_ and to calculate excitations of the negatively charged NV center in diamond [#Galli2]_.
An important advantage of QDET is its compatibility with quantum
algorithms as we explain in the following sections. It can be implemented for calculating
ground and excited states, as well as dynamic properties of materials. These make QDET a
powerful method for affordable quantum simulation of materials.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

"""

#############################################
# Theory
# ------
# QDET allows us to construct an effective Hamiltonian that describes the impurity
# subsystem while also accounting for its interaction with the environment, as follows
# environment as
#
# .. math::
#
#     H^{eff} = \sum_{ij}^{A} t_{ij}^{eff}a_i^{\dagger}a_j + \frac{1}{2}\sum_{ijkl}^{A} v_{ijkl}^{eff}a_i^{\dagger}a_{j}^{\dagger}a_ka_l,
#
# where :math:`t_{ij}^{eff}` and :math:`v_{ijkl}^{eff}` represent the effective one-body and
# two-body integrals, respectively, and :math:`ijkl` span over the orbitals inside the impurity.
# This Hamiltonian describes a simplified representation of the complex quantum system that is
# computationally tractable and properly captures the essential physics of the problem. The
# effective integrals :math:`t, v` are obtained
# from first-principles calculations [#Galli2]_.
#
# A QDET calculation is initiated obtaining a mean field approximation of the whole system
# using density functional theory (DFT). These calculations provide a set of orbitals
# which can be split into impurity and bath. An effective Hamiltonian is constructed from
# the impurity orbitals, which is subsequently solved by using either a high level
# classical method or a quantum algorithm. Let's implement these steps for an example!
#
# Implementation
# --------------
# We implement QDET to compute the excitation energies of a negatively charged nitrogen-vacancy
# defect in diamond.
#
# Mean field calculations
# ^^^^^^^^^^^^^^^^^^^^^^^
# We use DFT To obtain a mean field description of the whole system. The DFT
# calculations are performed with the QUANTUM ESPRESSO package. This requires downloading
# the pseudopotentials [#Modji]_ for each atomic species
# in the system from the QUANTUM ESPRESSO
# `database <https://www.quantum-espresso.org/pseudopotentials/>`_. We have carbon and nitrogen in
# our system which can be downloaded with
#
# .. code-block:: python
#
#    wget -N -q http://www.quantum-simulation.org/potentials/sg15_oncv/upf/C_ONCV_PBE-1.2.upf
#    wget -N -q http://www.quantum-simulation.org/potentials/sg15_oncv/upf/N_ONCV_PBE-1.2.upf
#
# Next, we need to create the input file for running QUANTUM ESPRESSO. The input file ``pw.in``
# contains information about the system and details of the DFT calculations. More details on
# how to construct the input file can be found in QUANTUM ESPRESSO
# `documentation <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`_ page.
#
# We can now perform the DFT calculations by running the executable code ``pw.x`` on the input file:
#
# .. code-block:: python
#
#    mpirun -n 2 pw.x -i pw.in > pw.out
#
# Identify the impurity
# ^^^^^^^^^^^^^^^^^^^^^
# Once we have obtained the mean field description, we can identify our impurity by finding
# the states that are localized in real space around the defect region. To that end, we compute the
# localization factor defined as:
#
# .. math::
#
#     L_n = \int_{V \in \ohm} d^3 r |\Psi_n^{KS}(r)|^2
#
# where :math:`V` is the identified volume including the impurity within the supercell volume :math:`\ohm` [#Galli2]_.
# We will use the `WEST <https://pubs.acs.org/doi/10.1021/ct500958p>`_ program to compute the localization factor. This requires creating the
# input file ``westpp.in`` as shown below.
#
# .. code-block:: python
#
#    westpp_control:
#      westpp_calculation: L # triggers the calculation of the localization factor
#      westpp_range:         # defines the range of states toe compute the localization factor
#      - 1                   # start from the first state
#      - 176                 # use all the 176 state
#      westpp_box:           # specifies the parameter of the box in atomic units for integration
#      - 6.19                #
#      - 10.19
#      - 6.28
#      - 10.28
#      - 6.28
#      - 10.28
#
# We can execute this calculation as
#
# .. code-block:: python
#
#    mpirun -n 2 westpp.x -i westpp.in > westpp.out
#
# This creates the file ``west.westpp.save/westpp.json``. Since computational resources required
# to run the calculation are large, the WEST output file needed for the next step can be
# directly downloaded as:
#
# .. code-block:: python
#
#    mkdir -p west.westpp.save
#    wget -N -q https://west-code.org/doc/training/nv_diamond_63/box_westpp.json -O west.westpp.save/westpp.json
#
# We can now plot the computed localization factor for each of the states:
#
# .. code-block:: python
#
#    import json
#    import numpy as np
#    import matplotlib.pyplot as plt
#
#    with open('west.westpp.save/westpp.json','r') as f:
#        data = json.load(f)
#
#    y = np.array(data['output']['L']['K000001']['local_factor'],dtype='f8')
#    x = np.array([i+1 for i in range(y.shape[0])])
#
#    plt.plot(x,y,'o')
#    plt.axhline(y=0.08,linestyle='--',color='red')
#
#    plt.xlabel('KS index')
#    plt.ylabel('Localization factor')
#
#    plt.show()
#
# From this plot, it is easy to see that Kohn-Sham orbitals can be catergorized as orbitals
# with low and high localization factor. For the purpose of defining an impurity, we need
# highly localized orbitals, so for this we set a cutoff of 0.06 and choose the orbitals
# that have a localization factor > 0.06 for our active space. We'll use these orbitals for
# the calculation of the parameters for the effective Hamiltonian in the following section.
#
# Effective Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^
# The next and probably most important steps in QDET is to define the effective one-body and
# two-body integrals for the impurity. The effective two-body integrals, :math:`v^{eff}` are computed
# first as matrix elements of the partially screened static Coulomb potential :math:`W_0^{R}`.
#
# .. math::
#
#     v_{ijkl}^{eff} = [W_0^{R}]_{ijkl},
#
# :math:`W_0^R`, results from screening the bare Coulomb potential, :math:`v`, with the reduced polarizability,
# :math:`P_0^R = P - P_{imp}`, where :math:`P` is the system's polarizability and :math:`P_{imp}` is the impurity's
# polarizability. Since solving the effective Hamiltonian
#  accounts for the exchange and correlation interactions between the active electrons, we remove these interactions from the
# the Kohn-Sham (KS) Hamiltonian, :math:`H^{KS}`, to avoid double counting them.
# Therefore, the one-body term :math:`t^{eff}` is obtained by subtracting
# from the Kohn-Sham Hamiltonian the double-counting (dc) term accounting for electrostatic and exchange-
# correlation interactions in the active space.
#
# .. math::
#
#     t_{ij}^{eff} = H_{ij}^{KS} - t_{ij}^{dc},
#
# In WEST, these parameters for the effective Hamiltonian are calculated by using the wfreq.x
# executable. The program will: (i) compute the quasiparticle energies, (ii) compute the
# partially screened Coulomb potential, and (iii) finally compute the parameters of the
# effective Hamiltonian. The input file for such a calculation is shown below:
#
# .. code-block:: python
#
#    wstat_control:
#      wstat_calculation: S
#      n_pdep_eigen: 512
#      trev_pdep: 0.00001
#
#    wfreq_control:
#      wfreq_calculation: XWGQH
#      macropol_calculation: C
#      l_enable_off_diagonal: true
#      n_pdep_eigen_to_use: 512
#      qp_bands: [87, 122, 123, 126, 127, 128]
#      n_refreq: 300
#      ecut_refreq: 2.0
#
# We now construct the effective Hamiltonian:
#
# .. code-block:: python
#
#    from westpy.qdet import QDETResult
#
#    mkdir -p west.wfreq.save
#    wget -N -q https://west-code.org/doc/training/nv_diamond_63/wfreq.json -O west.wfreq.save/wfreq.json
#
#    effective_hamiltonian = QDETResult(filename='west.wfreq.save/wfreq.json')
#
# The final step is to solve for this effective Hamiltonian using a high level method. We can
# use the WESTpy package as:
#
# .. code-block:: python
#
#    solution = effective_hamiltonian.solve()
#
# This effective Hamiltonian can be directly used with quantum algorithms in PennyLane
# once it is converted to a qubit Hamiltonian. Since WEST outputs two-electron integrals
# in chemists' notation, a conversion to the physicists' notation is essential for
# compatibility with PennyLane's framework. Here's how to construct the qubit Hamiltonian:
#
# .. code-block:: python
#
#    from pennylane.qchem import one_particle, two_particle, observable
#    import numpy as np
#
#    effective_hamiltonian = QDETResult(filename='west.wfreq.save/wfreq.json')
#
#    one_e, two_e = effective_hamiltonian.h1e, effective_hamiltonian.eri
#
#    t = one_particle(one_e[0])
#    v = two_particle(np.swapaxes(two_e[0][0], 1, 3))
#    qubit_op = observable([t, v], mapping="jordan_wigner")
#
# The ground state energy of the Hamiltonian is identical to the one obtained before.
#
# Conclusion
# ----------
# The quantum defect embedding theory is a novel framework for simulating strongly correlated
# quantum systems and has been successfully used for studying defects in solids. Applicability of
# QDET however is not limited to defects, it can be used for other systems where a strongly
# correlated subsystem is embedded in a weakly correlated environment. QDET is able to surpass the
# problem of correction of double counting of interactions within the active space faced by some
# other embedding theories such as DFT+DMFT.  Green's function based formulation of QDET ensures
# exact removal of double counting  corrections at GW level of theory, thus removing the
# approximation present in the initial DFT based formulation. This  formulation also helps capture
# the response properties and provides access to excited state properties. Another major advantage
# of QDET is the ease with which it can be used with quantum computers in a hybrid framework [#Baker]_.
# Therefore, we can conclude here that QDET is a powerful embedding approach for simulating complex
# quantum systems.
#
# References
# ----------
#
# .. [#Galli]
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
# .. [#Galli2]
#    Nan Sheng, Christian Vorwerk, *et al.*, "Green's function formulation of quantum defect embedding theory"
#    `Arxiv <https://arxiv.org/pdf/2203.05493>`__.
#
# .. [#Modji]
#    Modjtaba S. Zini, Alain Delgado, *et al.*, "Quantum simulation of battery materials using ionic pseudopotentials"
#    `ArXiv <https://arxiv.org/pdf/2302.07981>`__.
#
# .. [#Baker]
#    Jack S. Baker, Pablo A. M. Casares, *et al.*, "Simulating optically-active spin defects with a quantum computer"
#    `ArXiv <https://arxiv.org/pdf/2405.13115>`__.
#
# About the authors
# -----------------
#
