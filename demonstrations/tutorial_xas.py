r"""X-ray absorption spectroscopy simulation in the time domain
===========================================================

What will be the first industrially useful quantum algorithm to run on a fault-tolerant
quantum computer? This open question is one of the main focuses of the research team at Xanadu. A
potential answer to this question is simulating `X-ray absorption
spectroscopy <https://en.wikipedia.org/wiki/X-ray_absorption_spectroscopy>`__ (XAS), which can be used in
workflows to identify structural degradation mechanisms in material candidates for battery designs
🔋. This demo will show you how to implement an optimized simulation algorithm
developed in the paper `“Fast simulations of X-ray absorption spectroscopy for battery materials on
a quantum computer” <https://arxiv.org/abs/2506.15784>`__ [#Fomichev2025]_ in PennyLane.

First, we will discuss why simulating X-ray absorption spectroscopy is a promising application for
early quantum computers. Then we will explain the main steps in the simulation algorithm and how to
implement a simplified version in PennyLane.

 .. admonition:: Prerequisite understanding
    :class: note

    We will be using concepts that were introduced in other PennyLane demos, such as `Using PennyLane
    with PySCF and OpenFermion <https://pennylane.ai/qml/demos/tutorial_qchem_external>`__, `Initial
    state preparation for quantum
    chemistry <https://pennylane.ai/qml/demos/tutorial_initial_state_preparation>`__, and `How to build
    compressed double-factorized
    Hamiltonians <https://pennylane.ai/qml/demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians>`__.
    If you haven’t checked out those demos yet, it might be best to do so and then come back here 🔙.

Why simulate X-ray absorption spectroscopy?
-------------------------------------------

Lithium-excess materials are transition metal oxides that have been engineered to accommodate extra
Lithium atoms in their structural composition, designed as candidate materials for battery cathodes.
However, repeated charge-discharge cycles can alter their structure and reduce performance. One can
study these degraded materials using X-ray absorption spectroscopy, which directly probes local
structure by exciting tightly bound core electrons. This can be used to identify oxidation states in
materials because different elements and their oxidation states will absorb photons of different
energies. Characterizing the structures in the degraded cathode material can help in an iterative
development process, directing researchers towards better candidate materials.
Identification of the structures present in the degraded material is done by a process known as
“spectral fingerprinting”, where reference spectra of small molecular clusters are matched
to the experimental spectrum. A fast method of simulating reference spectra for use in
fingerprinting would be a crucial component in this iterative workflow for identifying
promising cathode materials.

.. figure:: ../_static/demonstration_assets/xas/fingerprinting.gif
   :alt: The reference spectra of molecular clusters are calculated, and then matched to an experimental spectrum.

Figure 1: *How simulation of X-ray absorption spectra can enable identification of oxidation states
in candidate battery materials.* Spectral fingerprinting can be used to identify constituent
structures of a material by decomposing experimental spectra into components calculated via
simulation on a quantum computer.

Simulating these spectra is a difficult task for classical computers – the highly correlated excited
states are difficult to compute classically, particularly for clusters with transition metals. 
However, the small number of electronic orbitals needed to simulate these small clusters make this
calculation well suited for early quantum computers, which can naturally handle the large amount of
correlation in the electron state.

Algorithm
---------

Simulating reference spectra requires calculating the observable of the experiment, which in this
case is the `absorption cross section <https://en.wikipedia.org/wiki/Absorption_cross_section>`__.
We will describe this quantity below and then explain how a *time-domain* simulation algorithm can
estimate it.

Absorption cross-section
~~~~~~~~~~~~~~~~~~~~~~~~

In XAS experiments, the absorption cross section as a function of the frequency of incident X-rays
:math:`\sigma_A(\omega)` is measured for a given material. This is related to the rate of absorption
of X-ray photons of various energies. For our situation, the electrons in the molecular cluster
start in a ground molecular state :math:`|I\rangle` with energy :math:`E_I`. This ground state will
be coupled to excited states :math:`|F\rangle` with energies :math:`E_F` through the action of the
dipole operator :math:`\hat m_\rho,` which represents the effect of the radiative field, where
:math:`\rho` is any of the Cartesian directions :math:`\{x,y,z\}`.

The absorption cross section is given by

.. math::  \sigma_A(\omega) = \frac{4 \pi}{3 \hbar c} \omega \sum_{F \neq I}\sum_{\rho=x,y,z} \frac{|\langle F|\hat m_\rho|I \rangle|^2 \eta}{((E_F - E_I)-\omega)^2 + \eta^2}\,,

where :math:`c` is the speed of light, :math:`\hbar` is Planck’s constant, and :math:`\eta` is the
line broadening -- set by the experimental resolution of the spectroscopy -- which is
typically around :math:`1` eV. Below is an illustration of an X-ray absorption spectrum.

.. figure:: ../_static/demonstration_assets/xas/example_spectrum.png
   :alt: Illustration of X-ray absorption spectrum with five peaks of varying positions and peak heights.
   :width: 50.0%
   :align: center

Figure 2: *Example X-ray absorption spectrum.* Illustration of how the peak positions
:math:`E_F - E_i`, widths :math:`\eta` and amplitudes
:math:`|\langle F | \hat m_\rho | I \rangle|^2` determine the spectrum.

The goal is to implement a quantum algorithm that can calculate this spectrum. However, instead of
computing the energy differences and state overlaps directly, we will be simulating the system in
the time domain, and then using a `Fourier
transform <https://en.wikipedia.org/wiki/Fourier_transform>`__ to obtain the spectrum in frequency space.

Quantum algorithm in the time-domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the initial state :math:`|I\rangle` and the dipole operator acting on the initial state
:math:`\hat m_\rho|I\rangle` can be determined classically, and we’ll demonstrate how to do that
later. With the initial state computed, we will use a mathematical trick called a *frequency-domain*
`Green’s function <https://en.wikipedia.org/wiki/Green%27s_function>`__ to determine the absorption
cross section. We can write the cross section as the imaginary part of the following Green’s
function

.. math:: \mathcal{G}_\rho(\omega) = \langle I|\hat m_\rho \frac{1}{\hat H -E_I -\omega +i\eta} \hat m_\rho |I\rangle\,.

Using a resolution of identity of the final states and simplifying, we end up with

.. math::  \mathrm{Im}(\mathcal{G_\rho(\omega)}) = -\sum_{F\neq I} \frac{|\langle F|\hat m_\rho|I\rangle|^2\eta}{(E_F- E_I -\omega)^2 +\eta^2} - \frac{|\langle I|\hat m_\rho|I\rangle|^2\eta}{\omega^2 +\eta^2}\,,

where the first term is clearly proportional to the absorption cross section. The second term is
zero if we centre the frame of reference for our molecular orbitals at the nuclear-charge weighted
centre for our molecular cluster of choice. Okay, so how do we determine
:math:`\mathcal{G_\rho(\omega)}`? If we are going to evaluate this quantity in a quantum register,
it will need to be normalized, so instead we are looking for

.. math::  G_\rho(\omega) = \eta \frac{\mathcal{G}_\rho(\omega)}{||\hat m_\rho | I \rangle ||^2} \,.

There are methods for determining this frequency-domain Green’s function directly [#Fomichev2024]_,
however, our algorithm will aim to estimate the discrete-time *time-domain* Green’s function
:math:`\tilde G(t_j)` at times :math:`t_j`, where :math:`j` is the time-step index.
:math:`G_\rho(\omega)` can then be calculated classically through the time-domain Fourier transform

.. math::  -\mathrm{Im}\,G_\rho(\omega) = \frac{\eta\tau}{2\pi} \sum_{j=-\infty}^\infty e^{-\eta |t_j|} \tilde G(t_j) e^{i\omega t_j}\,,

where :math:`\tau \sim \mathcal{O}(||\hat H||^{-1})` is the size of the time step. This step should be
small enough to resolve the largest frequency components that we are interested in, which correspond
to the final states with the largest energy. In practice, this is not the largest eigenvalue of
:math:`\hat H`, but simply the largest energy we want to show in the spectrum.

Now, where this all comes together is that the time-domain Green’s function can be determined using
the expectation value of the time-evolution operator (normalized)

.. math::  \tilde G_\rho(t_j) = \frac{\langle I|\hat m _\rho e^{- i\hat H t_j} \hat m_\rho |I\rangle}{|| \hat m_\rho |I\rangle ||^2}\,,

and this is something that can be easily done on a quantum computer! We can use a `Hadamard
test <https://en.wikipedia.org/wiki/Hadamard_test>`__ on the time evolution unitary to measure the
expectation value for each time :math:`t_j`. We will repeat this test a number of times :math:`N`
and take the mean of the results to get an estimate for :math:`G_\rho(t_j)`, which we can Fourier
transform to get the spectrum.

.. figure:: ../_static/demonstration_assets/xas/global_circuit.png
   :alt: Illustration of full Hadamard test circuit with state prep, time evolution and measurement.\
   :width: 70.0%
   :align: center

Figure 3: *Circuit for XAS simulation*. The algorithm is ultimately a Hadamard test circuit, and we
divide the steps of this into three components.

The circuit we will construct to determine the expectation values is shown above. It has three main
components:

- *State prep*, the state :math:`\hat m_\rho |I\rangle` is prepared in the quantum register,
  and an auxiliary qubit is prepared for controlled time evolution.
- *Time evolution*, the state is evolved under the electronic Hamiltonian.
- *Measurement*, the time-evolved state is measured to obtain statistics for the expectation value.

Let’s look at how to implement these steps in PennyLane. We will make extensive use of the
``qml.qchem`` module, as well as modules from `PySCF <https://pyscf.org/>`__. 

State preparation
-----------------

We need to classically determine the ground state :math:`|I\rangle`, and the dipole operator’s
action on that state :math:`\hat m_\rho |I\rangle`. For complicated molecular clusters, it's 
common to choose and consider only a subset of molecular orbitals and electrons in a calculation. 
This set of orbitals and electrons is known as the “active space” reduces the cost of performing 
calculations on complicated molecular instances, while hopefully still preserving the interesting 
features of the molecule. For this demo, we are going to use the simple :math:`\mathrm{H}_2` molecule,
and we will be trivially selecting the full space of orbitals and electrons for the calculation,
but the method demonstrated below will work for true subsets of more complicated molecules. 

Ground state calculation
~~~~~~~~~~~~~~~~~~~~~~~~

If you haven’t, check out the demo `“Initial state preparation for quantum
chemistry” <https://pennylane.ai/qml/demos/tutorial_initial_state_preparation>`__. We will be
expanding on that demo with code to import a state from the `multiconfigurational
self-consistent field <https://pyscf.org/user/mcscf.html>`__ (MCSCF) methods of PySCF, where we
restrict the set of active orbitals used in the calculation. 

We start by creating our molecule object using the `Gaussian type
orbitals <https://en.wikipedia.org/wiki/Gaussian_orbital>`__ module ``pyscf.gto``, and obtaining the
reduced Hartree-Fock molecular orbitals with the `self-consistent field
methods <https://pyscf.org/user/scf.html>`__ ``pyscf.scf``.
"""

from pyscf import gto, scf
import numpy as np

# Create a Mole object.
r = 0.71  # Bond length in Angstrom.
symbols = ["H", "H"]
geometry = np.array([[0.0, 0.0, -r / 2], [0.0, 0.0, r / 2]])
basis = "631g"
mol = gto.Mole(atom=zip(symbols, geometry), basis=basis, symmetry=None)
mol.build(verbose=0)

# Get the molecular orbitals.
hf = scf.RHF(mol)
hf.run(verbose=0)

######################################################################
# Since we will be using PennyLane for other aspects of this calculation, we want to make sure the
# molecular orbital coefficients are consistent between our PennyLane and PySCF calculations. To do
# this, we can obtain the molecular orbital coefficients from PennyLane using the ``hartree_fock.scf``
# method of ``qchem``, and change the coefficients in the ``hf`` instance to match.
#

import pennylane as qml

# Create a qml Molecule object.
mole = qml.qchem.Molecule(symbols, geometry, basis_name="6-31g", unit="angstrom")

# Run self-consistent fields method to get MO coefficients.
_, coeffs, _, _, _ = qml.qchem.hartree_fock.scf(mole)()

hf.mo_coeff = coeffs  # Change MO coefficients in hf object to PennyLane calculated values.

######################################################################
# Next, let’s define the active space of orbitals we will use for our calculation. This will be the
# number of orbitals ``n_orb_cas`` and the number of electrons ``n_electron_cas``. 
# For :math:`\mathrm{H}_2`, we can just use the full space, which is four orbitals and two electrons. 
# We will use a ``CASCI`` instance to calculate the ground state of our system with this 
# selected active space. The ``CASCI`` method in PySCF is equivalent to a full-configuration 
# interaction (FCI) procedure on a subset of molecular orbitals.
#

from pyscf import mcscf

# Define active space of (orbitals, electrons).
n_orb_cas, n_electron_cas = (4, 2)
ncore = (mol.nelectron - n_electron_cas) // 2

# Initialize CASCI instance of H2 molecule as mycasci.
mycasci = mcscf.CASCI(hf, ncas=n_orb_cas, nelecas=n_electron_cas)
mycasci.run(verbose=0)

# Calculate ground state, and omit small state components.
casci_state = mycasci.ci
casci_state[abs(casci_state) < 1e-6] = 0

######################################################################
# To implement this state as a PennyLane state vector, we need to convert the ``casci_state`` into a
# format that is easy to import into PennyLane. One way to do this is to use a sparse matrix
# representation to turn ``casci_state`` into a dictionary, and then use
# ``qml.qchem.convert.import_state`` to import into PennyLane. Here is how you can go about turning a
# full-configuration interaction matrix like ``casci_state`` into a dictionary.
#

from scipy.sparse import coo_matrix
from pyscf.fci.cistring import addrs2str

# Convert casci_state into a sparse matrix.
sparse_cascimatr = coo_matrix(casci_state, shape=np.shape(mycasci.ci), dtype=float)
row, col, dat = sparse_cascimatr.row, sparse_cascimatr.col, sparse_cascimatr.data

# Turn indices into strings.
n_orb_cas_a = mycasci.ncas
n_orb_cas_b = n_orb_cas_a
n_electron_cas_a, n_electron_cas_b = mycasci.nelecas
strs_row = addrs2str(n_orb_cas_a, n_electron_cas_a, row)
strs_col = addrs2str(n_orb_cas_b, n_electron_cas_b, col)

# Create the FCI matrix as a dict.
wf_casci_dict = dict(zip(list(zip(strs_row, strs_col)), dat))

######################################################################
# Lastly, we will use the helper function ``_sign_chem_to_phys`` to adjust the sign of state
# components to match what they should be for the PennyLane orbital occupation number ordering. Then,
# we can import the state to PennyLane using ``_wf_dict_to_statevector``.
#
# .. admonition:: Chemist's and physicist's notation for spin orbitals
#     :class: note
#
#     In general, states from computation chemistry workflows will have spin orbitals ordered
#     in chemist's notations, such that all of one spin is on the left, and the other on the right. 
#     PennyLane uses the physicist's notation, where the spatial orbitals are ordered, and the spins 
#     alternate up and down. When changing a state from one convention to the next, the sign of some 
#     state amplitudes needs to change to adhere to the Fermionic anticommutation rules. The helper 
#     function ``_sign_chem_to_phys`` does this sign adjustment for us.
#

from pennylane.qchem.convert import _sign_chem_to_phys, _wfdict_to_statevector

# Adjust sign of state components to match physicist's notation.
wf_casci_dict = _sign_chem_to_phys(wf_casci_dict, n_orb_cas)

# Convert dictionary to Pennylane state vector.
wf_casci = _wfdict_to_statevector(wf_casci_dict, n_orb_cas)

######################################################################
# Dipole operator action
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The electromagnetic field of the X-rays couples electronic states through the `dipole
# operator <https://en.wikipedia.org/wiki/Transition_dipole_moment>`__. The action of this operator is
# implemented in PennyLane as ``qml.qchem.dipole_moment``. We can calculate that operator, convert it
# to a matrix, and apply it to our initial state :math:`|I\rangle` to obtain :math:`\hat m_\rho|I\rangle`.
#
# To generate this operator, we have to specify which molecular orbitals are in our active space. We
# can obtain the indices of the included and excluded orbitals using ``qml.qchem.active_space`` to
# obtain the lists ``active`` and ``core``, respectively.
#
# The action of the dipole operator will be split into the three cartesian directions
# :math:`\{x, y, z\}`, which we will loop over to obtain the states :math:`\hat m_{\{x,y,z\}}|I\rangle`.
#

# Solve for active space.
core, active = qml.qchem.active_space(
    mole.n_electrons, mole.n_orbitals, 
    active_electrons=n_electron_cas, 
    active_orbitals=n_orb_cas
)

m_rho = qml.qchem.dipole_moment(mole, cutoff=1e-8, core=core, active=active)()
rhos = range(len(m_rho))  # [0, 1, 2] are [x, y, z].

wf_dipole = []
dipole_norm = []

# Loop over cartesian coordinates and calculate m_rho|I>.
for rho in rhos:
    dipole_matrix_rho = qml.matrix(m_rho[rho], wire_order=range(2 * n_orb_cas))
    wf = dipole_matrix_rho.dot(wf_casci)

    if np.allclose(wf, np.zeros_like(wf)):  # If wf is zero, then set norm as zero.
        wf_dipole.append(wf)
        dipole_norm.append(0)

    else:  # Normalize the wavefunction.
        dipole_norm.append(np.linalg.norm(wf))
        wf_dipole.append(wf / dipole_norm[rho])

######################################################################
# .. admonition:: Caution! Wire ordering
#     :class: note
#
#     When converting the operator ``m_rho`` to the matrix ``dipole_matrix_rho``,
#     the full set of wires need to be specified, otherwise the matrix may not have the
#     right dimension (if, for example, the operator is zero along any cartesian direction).
#
# Let’s prepare the circuit that will initialize our qubit register with this state.
# We will need :math:`2 n_\mathrm{cas}` wires, which is twice our full space since we need
# to account for spin. We will also add one auxiliary wire for the measurement circuit,
# which we will prepare as the 0 wire with an applied Hadamard gate.
#

import pennylane as qml

device_type = "lightning.qubit"

# Initialization circuit for m_rho|I>.
dev_prop = qml.device(device_type, wires=int(2*n_orb_cas) + 1, shots=None)


@qml.qnode(dev_prop)
def initial_circuit(wf):
    # Dipole wavefunction preparation.
    qml.StatePrep(wf, wires=dev_prop.wires.tolist()[1:])
    qml.Hadamard(wires=0)
    return qml.state()


######################################################################
# .. note::
# 
#     To make guarantee that :math:`\langle I|\hat m_\rho|I\rangle` is zero, we require that the ``Mole`` object’s
#     nuclear-charge-weighted centre is at the origin. Note this is true from our construction, since 
#     the geometry was defined to be symmetric about the origin, but I want to emphasize the importance of this condition.
#
# Time Evolution
# --------------
#
# Next we will discuss how to prepare the electronic Hamiltonian for use in the time evolution of the
# Hadamard-test. We will double-factorize and compress the Hamiltonian to obtain an approximation of the 
# Hamiltonian that can be easily fast-forwarded in a Trotter product formula.
#
# If you haven’t yet, go read the demo `“How to build compressed double-factorized
# Hamiltonians” <https://pennylane.ai/qml/demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians>`__,
# because that is exactly what we are going to do!
#
# Electronic Hamiltonian
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Our electronic Hamiltonian is
#
# .. math::  H = E + \sum_{p,q=1}^N \sum_{\gamma\in\{\uparrow,\downarrow\}} (p|\kappa|q) a^\dagger_{p\gamma}a_{q\gamma} + \frac12 \sum_{p,q,r,s=1}^N\sum_{\gamma,\beta\in\{\uparrow,\downarrow\}} (pq|rs) a^\dagger_{p\gamma}a_{q\gamma} a^\dagger_{r\beta}a_{s\beta} \,,
#
# where :math:`a^{(\dagger)}_{p\gamma}` is the annihilation (creation) operator for a molecular
# orbital :math:`p` and spin :math:`\gamma`, :math:`E` is the core constant, :math:`N` is the number
# of spatial orbitals, and :math:`(p|\kappa|q)` and :math:`(pq|rs)` are the one- and two-electron
# integrals, respectively [#Cohn2021]_.
#
# The core constant and the one- and two-electron integrals can be computed in PennyLane using
# functions from ``qml.qchem.hartree_fock``.
#

# Calculate electron integrals.
core_constant, one, two = qml.qchem.electron_integrals(mole, core=core, active=active)()
core_constant = core_constant[0]

######################################################################
# We will have to convert these to chemist’s notation [#Sherrill2005]_.
#

# To chemist notation.
two_chemist = np.einsum("prsq->pqrs", two)
one_chemist = one - np.einsum("pqrr->pq", two) / 2.0

######################################################################
# Next, we will double factorize and compress the Hamiltonian's one- and two-electron terms. 
# This approximates the Hamiltonian as [#Yen2021]_ [#Cohn2021]_ 
#
# .. math::  H_\mathrm{CDF} = E + \sum_{\gamma\in\{\uparrow,\downarrow\}} U_\gamma^{(0)} \left(\sum_p Z_p^{(0)} a_{\gamma,p}^\dagger a_{\gamma, p}\right) U_\mathrm{\gamma}^{(0)\,\dagger} + \sum_\ell^L \sum_{\gamma,\beta\in\{\uparrow,\downarrow\}} U_\mathrm{\gamma, \beta}^{(\ell)} \left( \sum_{pq} Z_{pq}^{(\ell)} a_{\gamma, p}^\dagger a_{\gamma, p} a_{\beta,q}^\dagger a_{\beta, q}\right) U_{\gamma, \beta}^{(\ell)\,\dagger} \,,
#
# where each one-electron integral is approximated by a matrix :math:`Z^{(0)}` surrounded by
# single-particle rotation matrices :math:`U^{(0)}` which diagonalize :math:`Z^{(0)}`. Each
# two-electron integral is approximated by a sum of :math:`L` of these rotation and diagonal matrix
# terms, indexed as :math:`(\ell)`. 
#
# We can compress and double-factorize the two-electron integrals using PennyLane’s
# ``qchem.factorize`` function, with ``compressed=True``. We will set :math:`L` as the number of
# orbitals in our active space. The ``Z`` and ``U`` output here are arrays of :math:`L` fragment matrices
# with dimension :math:`n_\mathrm{cas} \times n_\mathrm{cas}`.
#

# Factorize hamiltonian, producing matrices.
L = n_orb_cas  # Usually L is on the order of n_orb_cas.
_, Z, U = qml.qchem.factorize(two_chemist, compressed=True, num_factors=L)

print("Shape of the factors: ")
print("two_chemist", two_chemist.shape)
print("U", U.shape)
print("Z", Z.shape)

# Compare factorized two-electron terms to the originals.
approx_two_chemist = qml.math.einsum("tpk,tqk,tkl,trl,tsl->pqrs", U, U, Z, U, U)
assert qml.math.allclose(approx_two_chemist, two_chemist, atol=1e-2)

######################################################################
# Note there are some terms in this decomposition that are exactly diagonalizable, and can be added to
# the one-electron terms to simplify the simulation. We call these the “one-electron extra” terms and
# add them to the one-electron integrals, using ``np.linalg.eigh`` to diagonalize them into the matrix
# :math:`Z^{(0)}` with the rotation matrix :math:`U^{(0)}`.
#

# Calculate the one-electron extra.
Z_prime = np.stack([np.diag(np.sum(Z[i], axis=-1)) for i in range(Z.shape[0])], axis=0)
one_electron_extra = np.einsum("tpk,tkk,tqk->pq", U, Z_prime, U)

# Diagonalize the one-electron integral matrix while adding the one-electron extra.
eigenvals, U0 = np.linalg.eigh(one_chemist + one_electron_extra)
Z0 = np.diag(eigenvals)

######################################################################
# Constructing the time-propagation circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The main work of our algorithm will be to apply our Hamiltonian terms as a Trotter product in a
# time-evolution operator, and measure the expectation value of that time evolution for various times.
# Let’s start by writing functions that implement the time evolution for each Hamiltonian term, which
# will be called by our Trotter circuit. 
#
# The trick when implementing a double-factorized Hamiltonian is to use `Thouless’s
# theorem <https://joshuagoings.com/assets/Thouless_theorem.pdf>`__ [#Thouless1960]_ to apply the
# single-particle basis rotations :math:`U^{(\ell)}`. The Jordan-Wigner transform can then
# implement the number operators :math:`a^\dagger_p a_p = n_{p}` as Pauli-Z rotations, via 
# :math:`n_p = (1-\sigma_{z,p})/2`. Note the :math:`1/2` term will change the global phase, 
# and we will have to keep track of that carefully. 
# Below is an illustration of the circuit we will use to implement the one- and two-eletron 
# terms in our factorized Hamiltonian.
#
# .. figure:: ../_static/demonstration_assets/xas/UZU_circuits.png
#    :alt: One- and two-electron basis rotation and Pauli-Z rotation circuits.
#    :width: 80.0%
#    :align: center
#
# Figure 4: One- and two-electron term implementations in the time-evolution circuit (ignoring global
# phases). Basis rotations are applied to both spin sections of the register.
#
# We can use ``qml.BasisRotation`` to generate a `Givens
# decomposition <https://pennylane.ai/qml/demos/tutorial_givens_rotations>`__ for the single-body
# basis rotation determined by :math:`U^{(\ell)}`. We will have to do this for both spin-halves of the
# register.
#


def U_rotations(U, control_wires):
    """Circuit implementing the basis rotations of the CDF decomposition."""
    U_spin = qml.math.kron(U, qml.math.eye(2))  # Apply to both spins.
    qml.BasisRotation(
        unitary_matrix=U_spin, wires=[int(i + control_wires) for i in range(2 * n_orb_cas)]
    )


######################################################################
# Next we write a function to perform the Z rotations. Controlled arbitrary-angle rotations
# are expensive. To reduce the cost of having to implement many controlled Z rotations at angles
# determined by the matrices :math:`Z^{(\ell)}`, we instead implement *uncontrolled* Z rotations
# sandwiched by CNOT gates.
#
# .. figure:: ../_static/demonstration_assets/xas/double_phase_trick.png
#    :alt: Diagram showing that a controlled-Z rotation of 2 theta is equivalent to a Z rotation of theta sandwiched by CNOT gates.
#
# Figure 5: Double-phase trick to decompose expensive controlled-Z rotations into an uncontrolled-Z
# rotation sandwiched by CNOT gates.
#
# For the one-electron terms, we loop over spin and orbital index, and apply the Z rotations using
# this double-phase trick. The two-electron terms are implemented the same way, except the two-qubit
# rotations use ``qml.MultiRZ``.
#

from itertools import product


def Z_rotations(Z, step, is_one_electron_term, control_wires):
    """Circuit implementing the Z rotations of the CDF decomposition."""
    if is_one_electron_term:
        for sigma in range(2):
            for i in range(n_orb_cas):
                qml.ctrl(
                    qml.X(wires=int(2*i + sigma + control_wires)),
                    control=range(control_wires),
                    control_values=0,
                )
                qml.RZ(-Z[i, i] * step / 2, wires=int(2*i + sigma + control_wires))
                qml.ctrl(
                    qml.X(wires=int(2*i + sigma + control_wires)),
                    control=range(control_wires),
                    control_values=0,
                )
        globalphase = np.sum(Z) * step  # Phase from 1/2 term in Jordan-Wigner transform.

    else:  # It's a two-electron term.
        for sigma, tau in product(range(2), repeat=2):
            for i, k in product(range(n_orb_cas), repeat=2):
                if i != k or sigma != tau:  # Skip the one-electron correction terms.
                    qml.ctrl(qml.X(wires=int(2*i + sigma + control_wires)),
                            control=range(control_wires), control_values=0)
                    qml.MultiRZ(Z[i, k] / 8.0 * step,
                        wires=[int(2*i + sigma + control_wires),
                               int(2*k + tau + control_wires)])
                    qml.ctrl(qml.X(wires=int(2 * i + sigma + control_wires)),
                        control=range(control_wires), control_values=0)
        globalphase = np.trace(Z)/4.0*step - np.sum(Z)*step + np.sum(Z)*step/2.0

    qml.PhaseShift(-globalphase, wires=0)


######################################################################
# .. note::
#    For a derivation of the global phase for the two-electron terms, 
#    see Appendix A in [#Fomichev2025]_.
#
# Now that we have functions for the complicated terms of our Hamiltonian, we can define our Trotter
# step. The function will implement the :math:`U` rotations and :math:`Z` rotations, and adjust the
# global phase from the core constant term. By tracking the last :math:`U` rotation used, we can
# implement two consecutive rotations at once as :math:`V^{(\ell)} = U^{(\ell-1)}(U^{(\ell)})^T`,
# halving the number of rotations required per Trotter step.
#
# Below, we define a function ``LieTrotter`` which applies the rotations for the one- and two-electron
# terms in one order, but can also reverse the order. This can save another rotation step when we
# implement consecutive Trotter steps in higher-order Trotter schemes. At the end of the step, the
# core constant adjusts a global phase.
#


def LieTrotter(step, prior_U, final_rotation, reverse=False):
    """Implements a first-order Trotterized circuit for the CDF."""
    # Combine the one- and two-electron matrices.
    _U0 = np.expand_dims(U0, axis=0)
    _Z0 = np.expand_dims(Z0, axis=0)
    _U = np.concatenate((_U0, U), axis=0)
    _Z = np.concatenate((_Z0, Z), axis=0)

    num_two_electron_terms = U.shape[0]  # Number of fragments \ell.
    is_one_body = np.array([True] + [False] * num_two_electron_terms)
    order = list(range(len(_Z)))

    if reverse:
        order = order[::-1]

    for term in order:
        U_rotations(prior_U @ _U[term], 1)
        Z_rotations(_Z[term], step, is_one_body[term], 1)
        prior_U = _U[term].T

    if final_rotation:
        U_rotations(prior_U, 1)

    # Global phase adjustment from core constant.
    qml.PhaseShift(-core_constant * step, wires=0)

    return prior_U


######################################################################
# Our function ``trotter_circuit`` implements a second-order Trotter step, returning a ``QNode``. The
# returned circuit applies ``StatePrep`` to prepare the register in the previous quantum state, and
# then two ``LieTrotter`` evolutions for time ``step/2`` so that the total step size is ``step``.
#


def trotter_circuit(dev, state, step):
    """Implements a second-order Trotterized circuit for the CDF."""
    qubits = dev.wires.tolist()

    def circuit():
        # State preparation -- set as previous iteration final state.
        qml.StatePrep(state, wires=qubits)

        # Main body of the circuit.
        prior_U = np.eye(n_orb_cas)  # No initial prior U, so set as identity matrix.
        prior_U = LieTrotter(step / 2, prior_U=prior_U, final_rotation=False, reverse=False)
        prior_U = LieTrotter(step / 2, prior_U=prior_U, final_rotation=True, reverse=True)

        return qml.state()

    return qml.QNode(circuit, dev)


######################################################################
# Measurement
# -----------
#
# To measure the expectation value of the time-propagated state, we use a Hadamard test circuit. This
# uses ``qml.StatePrep`` to set the state as it was returned by the time evolution, and then measures
# both the real and imaginary expectation values using ``PauliX`` and ``PauliY``, respectively.
#


def meas_circuit(state):
    qml.StatePrep(state, wires=range(int(2*n_orb_cas) + 1))
    # Measure in PauliX/PauliY to get the real/imaginary parts.
    return [qml.expval(op) for op in [qml.PauliX(wires=0), qml.PauliY(wires=0)]]


######################################################################
# We can only obtain both real and imaginary expectation values in a *simulated* circuit. An
# actual implementation would have to select real or imaginary by inserting a phase gate, like in the
# circuit below.
#
# .. figure:: ../_static/demonstration_assets/xas/hadamard_test_circuit.png
#    :alt: Hadamard test circuit with optional S-dagger gate on the auxiliary qubit.
#    :width: 70.0%
#    :align: center
#
# Figure 6: *Hadamard test circuit to measure expectation value of time-evolution operator*. With the
# phase gate :math:`S^\dagger` present (absent), this gives the real (imaginary) part of the
# time-domain Green’s function :math:`\tilde G(\tau j)`.
#
# Run Simulation
# --------------
# Let’s define the simulation parameters we are going to use. This includes:
#
#  - The Lorentzian width :math:`\eta` of the spectrum peaks, representing the experimental resolution.
#  - The time step :math:`\tau`, which should be small enough to resolve the largest frequency components
#    we want to determine.
#  - The maximum number of time steps :math:`j_\mathrm{max}`, which sets the largest evolution time.
#    This should be large enough so that we can distinguish between the small frequency components in
#    our spectrum.
#  - The total number of shots we will use to obtain statistics for the expectation value after the time
#    evolution.
#

eta = 0.05  # In Hartree energy units (Ha).
H_norm = 1.5  # Maximum final state eigenvalue used to determine tau.
tau = np.pi / (2 * H_norm)  # Time step, set by largest relevant eigenvalue.
jmax = 40  # Max number of time steps.
total_shots = 500 * 2 * jmax  # Total number of shots for expectation value statistics.

jrange = np.arange(1, 2 * int(jmax) + 1, 1)
time_interval = tau * jrange

######################################################################
# Minimizing the number of shots we require to obtain the necessary expectation value statistics will
# improve the efficiency of our algorithm. One way to do this is to employ a sampling distribution
# that takes advantage of the decaying Lorentzian kernel, exponentially reducing the shot allocation
# for longer evolution times [#Fomichev2025]_. This is implemented below by creating ``shots_list``,
# which distributes the ``total_shots`` among the time steps, weighted exponentially by the Lorentzian width. The
# parameter :math:`\alpha` can adjust this weighting, s.t. for :math:`\alpha > 1` there is more weight
# at shorter times.
#


def L_j(t_j):
    """Time-dependent shot distribution."""
    return np.exp(-eta * t_j)


alpha = 1.1  # Tunable kernel weighting.

# Normalization factor so total shots is as defined.
A = np.sum([L_j(alpha * t_j) for t_j in time_interval])  

# Kernel-aware list of shots for each time step.
shots_list = [int(round(total_shots * L_j(alpha * t_j) / A)) for t_j in time_interval]

######################################################################
# Finally, we can run the simulation to determine the expectation values at each time step, which are
# related to the time-domain Green’s function.
#

expvals = np.zeros((2, len(time_interval)))  # Results list initialization.

# Loop over cartesian coordinate directions.
for rho in rhos:

    if dipole_norm[rho] == 0:  # Skip if no excited states are coupled.
        continue

    # Initialize state m_rho|I> (including the auxiliary qubit).
    state = initial_circuit(wf_dipole[rho])

    # Perform time steps.
    for i in range(0, len(time_interval)):

        circuit = trotter_circuit(dev=dev_prop, state=state, step=tau)

        # Define measurement circuit device with shots.
        shots = shots_list[i]  # Kernel-aware number of shots.
        dev_est = qml.device(device_type, wires=int(2*n_orb_cas) + 1, shots=shots)

        # Update state and then measure expectation values.
        state = circuit()
        measurement = qml.QNode(meas_circuit, dev_est)(state)

        expvals[:, i] += dipole_norm[rho]**2 * np.array(measurement).real

######################################################################
# Plotting the time-domain output, we see there is one clear frequency, so we will expect one peak in
# our spectrum.
#

import matplotlib.pyplot as plt

plt.style.use("pennylane.drawer.plot")

fig = plt.figure(figsize=(6.4, 2.4))
ax = fig.add_axes((0.15, 0.3, 0.8, 0.65))  # Leave space for caption.
ax.plot(range(len(expvals[0, :])), expvals[0, :], label="Real")
ax.plot(range(len(expvals[1, :])), expvals[1, :], label="Imaginary", linestyle="--")
ax.set(xlabel=r"$\mathrm{Time step}, j$", ylabel=r"$\mathrm{Expectation Value}")
fig.text(0.5, 0.05,
    "Figure 7. Time-domain output of algorithm.",
    horizontalalignment="center",
    size="small", weight="normal")
ax.legend()
plt.show()

######################################################################
# Since the real and imaginary components of the time-domain Green’s function are determined
# separately, we can calculate the Fourier transform like
#
# .. math::  -\mathrm{Im}\,G_\rho(\omega) = \frac{\eta\tau}{2\pi}\left(1 + 2\sum_{j=1}^{j_\mathrm{max}}\left[ \mathbb{E}\left(\mathrm{Re}\,\tilde G(\tau j)\right)\mathrm{cos}(\tau j \omega) - \mathbb{E}\left(\mathrm{Im}\,\tilde G(\tau j)\right) \mathrm{sin}(\tau j \omega)\right]\right) \,,
#
# where :math:`\mathbb{E}` is the expectation value. We do this below, but also multiply the
# normalization factors over to the right side.
#

L_js = L_j(time_interval)

f_domain_Greens_func = (
    lambda w: tau/(2*np.pi) * (np.sum(np.array(dipole_norm)**2) 
            + 2*np.sum(L_js * (expvals[0, :] * np.cos(time_interval * w)
            - expvals[1, :] * np.sin(time_interval * w)))))

wgrid = np.linspace(-2, 5, 10000)  # Frequency array for plotting.
w_min, w_step = wgrid[0], wgrid[1] - wgrid[0]

spectrum = np.array([f_domain_Greens_func(w) for w in wgrid])

######################################################################
# Since our active space for :math:`\mathrm{H}_2` is small, we can easily calculate a classical spectrum for
# comparison. We do this using the ``mycasci`` instance that we used to determine the ground state,
# but instead solve for more states by increasing the number of roots in the ``fcisolver``. 
# This will give us the energies of the coupled final staes. We can also calculate the 
# transition density matrix in the molecular orbital basis between those states and the initial state,
# :math:`\langle F| \hat m_\rho |I \rangle`. Finally, we can compute the absorption cross section directly.
# 

# Use CASCI to solve for excited states.
mycasci.fcisolver.nroots = 10  # Compute the first 10 states.
mycasci.run(verbose=0)
mycasci.e_tot = np.atleast_1d(mycasci.e_tot)

# Ground state energy.
E_i = mycasci.e_tot[0]

# Determine the dipole integrals using atomic orbitals from RHF object.
dip_ints_ao = hf.mol.intor("int1e_r_cart", comp=3)  # In atomic orbital basis.
mo_coeffs = coeffs[:, ncore : ncore + n_orb_cas]

# Convert to molecular orbital basis.
dip_ints_mo = np.einsum("ik,xkl,lj->xij", mo_coeffs.T, dip_ints_ao, mo_coeffs)


def makedip(ci_id):
    # Transition density matrix in molecular orbital basis.
    t_dm1 = mycasci.fcisolver.trans_rdm1(
        mycasci.ci[0], mycasci.ci[ci_id], n_orb_cas, n_electron_cas
    )
    # Transition dipole moments.
    return np.einsum("xij,ji->x", dip_ints_mo, t_dm1)


F_m_Is = np.array([makedip(i) for i in range(len(mycasci.e_tot))])

# Absorption cross section.
spectrum_classical_func = lambda E: (1 / np.pi) * np.sum(
                [np.sum(np.abs(F_m_I)**2) * eta / ((E - e)**2 + eta**2)
                    for (F_m_I, e) in zip(F_m_Is, mycasci.e_tot)])

spectrum_classical = np.array([spectrum_classical_func(w) for w in wgrid])

######################################################################
# Let’s plot and compare the classical and quantum spectra.
#

fig = plt.figure(figsize=(6.4, 4))
ax = fig.add_axes((0.15, 0.20, 0.80, 0.72))  # Make room for caption.

ax.plot(wgrid - E_i, spectrum, label="quantum")
ax.plot(wgrid - E_i, spectrum_classical, "--", label="classical")
ax.set_xlabel(r"$\mathrm{Energy}, \omega\ (\mathrm{Ha})$")
ax.set_ylabel(r"$\mathrm{Absorption\ (arb.)}$")
ax.legend()
fig.text(0.5, 0.05,
    r"Figure 8: $\mathrm{H}_2$ XAS spectrum calculation.",
    horizontalalignment="center",
    size="small", weight="normal")
plt.show()

######################################################################
# Core-valence separation approximation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For larger molecular instances, it may be valuable to restrict the terms coupled by the dipole
# operator to only include those of relevance for XAS, which are final states where a *core*
# electron is excited, i.e. there exists a hole in the core orbitals. These are known as
# core-excited states, and lie significantly above the valence-excited states in energy. Typically the
# frequency range is focused on a target atom in a molecular cluster, and also near a transition
# energy, such as targeting core :math:`1s` electrons.
#
# Atomic species and oxidations states will determine the energy difference between states with
# different principle quantum numbers, and this difference will show as a peak in spectroscopy, known
# as the *absorption edge*. Focusing spectroscopy near this edge for :math:`1s` to :math:`2p` is
# called the :math:`K`-edge, and in general X-ray absorption near-edge spectroscopy is known as XANES.
# By applying the core-valence separation approximation, we can force our calculation to stay in the
# XANES region.
#
# .. figure:: ../_static/demonstration_assets/xas/core_valence.png
#    :alt: Energy diagram with X-rays exciting core electrons to high valence energies, and UV and visible radiation only excite electrons already in valence orbitals.
#    :width: 50.0%
#    :align: center
#
# Figure 9: *Core-valence separation.* A much larger amount of energy is required to excite core
# electrons into valence orbitals compared to electrons already in low-lying valence orbitals. Since
# XAS targets core electrons, we can ignore valence-excitation matrix elements in our calculations.
#
# Further Optimizations
# ~~~~~~~~~~~~~~~~~~~~~
#
# There are more optimizations for this algorithm that are included in the paper [#Fomichev2025]_
# that we did not implement in the above code. One could further optimize the compressed
# double-factorized Hamiltonian by applying a block-invariant symmetry shift (BLISS) [#Loaiza2023]_
# to the Hamiltonian prior to compression. This is already detailed in the `demo on CDF
# Hamiltonians <https://pennylane.ai/qml/demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians>`__.
#
# Another optimization is to use a randomized second-order Trotter formula for the time evolution. As
# discussed in Ref. [#Childs2019]_, deterministic product formulas have error that scales with the
# commutators of the Hamiltonian terms. One could instead use all permutations of the Hamiltonian
# terms, such that the commutator errors cancel. However, the average of all permutations is not
# unitary in general. To circumvent this, one can randomly choose a Hamiltonian term ordering, which
# can give a good approximation to the desired evolution.
#
# More efficient methods of simulating XAS may be discovered in the near future, which could make this
# application even more viable as a use for early fault-tolerant quantum computers.
#
# Conclusion
# ----------
#
# In this tutorial, we have implemented a simplified version of the algorithm as presented in
# [#Fomichev2025]_. The algorithm represents a culmination of many optimizations for time-evolving an
# electronic Hamiltonian. We’ve also discussed how XAS is a promising candidate for early
# fault-tolerant quantum computers due to its low qubit overhead but high amount of correlations in
# the state space.
#
# *Acknowledgements*: The author thanks Stepan Fomichev and Pablo A. M. Casares for providing the code
# used in [#Fomichev2025]_, which was a basis for the simplified implementation demonstrated here.
#
# References
# ----------
#
# .. [#Fomichev2025]
#
#    Stepan Fomichev, Pablo A. M. Casares, Jay Soni, Utkarsh Azad, Alexander Kunitsa, Arne-Christian
#    Voigt, Jonathan E. Mueller, and Juan Miguel Arrazola, “Fast simulations of X-ray absorption spectroscopy
#    for battery materials on a quantum computer”. `arXiv preprint arXiv:2506.15784
#    (2025) <https://arxiv.org/abs/2506.15784>`__.
#
# .. [#Fomichev2024]
#
#    Stepan Fomichev, Kasra Hejazi, Ignacio Loaiza, Modjtaba Shokrian Zini, Alain Delgado, Arne-Christian
#    Voigt, Jonathan E. Mueller, and Juan Miguel Arrazola, “Simulating X-ray absorption spectroscopy of
#    battery materials on a quantum computer”. `arXiv preprint arXiv:2405.11015
#    (2024) <https://arxiv.org/abs/2405.11015>`__.
#
# .. [#Loaiza2023]
#
#    Ignacio Loaiza and Artur F Izmaylov, “Block-invariant symmetry shift: Preprocessing technique for
#    second-quantized Hamiltonians to improve their decompositions to linear combination of unitaries”.
#    `J. Chem. Theory Comput. 19, 22, 8201–8209 (2023) <https://doi.org/10.1021/acs.jctc.3c00912>`__.
#
# .. [#Yen2021]
#
#    Tzu-Ching Yen and Artur F. Izmaylov, “Cartan subalgebra approach to efficient measurements of
#    quantum observables”, `PRX Quantum 2, 040320
#    (2021) <https://doi.org/10.1103/PRXQuantum.2.040320>`__.
#
# .. [#Cohn2021]
#
#    Jeffrey Cohn, Mario Motta, and Robert M. Parrish, “Quantum filter diagonalization with compressed
#    double-factorized Hamiltonians”. `PRX Quantum 2, 040352
#    (2021) <https://doi.org/10.1103/PRXQuantum.2.040352>`__.
#
# .. [#Childs2019]
#
#    Andrew M. Childs, Aaron Ostrander, and Yuan Su, “Faster quantum simulation by randomization”. `Quantum
#    3, 182 (2019) <https://doi.org/10.22331/q-2019-09-02-182>`__.
#
# .. [#Sherrill2005]
#
#    C. David Sherrill, `“Permutational symmetries of one- and two-electron integrals”
#    (2005) <https://vergil.chemistry.gatech.edu/static/content/permsymm.pdf>`__.
#
# .. [#Thouless1960]
#
#    David J. Thouless, “Stability conditions and nuclear rotations in the Hartree-Fock theory”. `Nuclear
#    Physics, 21, 225-232 (1960) <https://doi.org/10.1016/0029-5582(60)90048-1>`__.
#

######################################################################
# About the author
# ----------------
#
