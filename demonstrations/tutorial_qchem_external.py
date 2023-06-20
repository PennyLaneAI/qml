r"""

Using PennyLane with PySCF and OpenFermion
==========================================

.. meta::
    :property="og:description": Learn how to integrate external quantum chemistry libraries with PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_external_libs.png


.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry

*Author: Soran Jahangiri — Posted: 3 January 2023.*

The quantum chemistry module in PennyLane, :mod:`qml.qchem  <pennylane.qchem>`, provides built-in
methods to compute molecular integrals, solve Hartree-Fock equations, and construct
`fully-differentiable <https://pennylane.ai/qml/demos/tutorial_differentiable_HF.html>`_ molecular
Hamiltonians. However, there are many other interesting and widely used quantum chemistry libraries out there. Instead of reinventing the wheel, PennyLane lets you to take advantage of various external resources and libraries to build upon existing research. In this demo we will show you how to integrate PennyLane with `PySCF <https://github.com/sunqm/pyscf>`_ and
`OpenFermion <https://github.com/quantumlib/OpenFermion>`_ to compute molecular integrals and
construct molecular Hamiltonians.

Building molecular Hamiltonians
-------------------------------
In PennyLane, Hamiltonians for quantum chemistry are built with the
:func:`~.pennylane.qchem.molecular_hamiltonian` function by specifying a backend for solving the
Hartree–Fock equations. The default backend is the differentiable Hartree–Fock solver of the
:mod:`qml.qchem <pennylane.qchem>` module. A molecular Hamiltonian can also be constructed with a
non-differentiable backend that uses the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin, interfaced with the
electronic structure package `PySCF <https://github.com/sunqm/pyscf>`_. This
backend can be selected by setting ``method='pyscf'`` in
:func:`~.pennylane.qchem.molecular_hamiltonian`. This requires the ``OpenFermion-PySCF``
plugin to be installed by the user with the following:

.. code-block:: bash

   pip install openfermionpyscf

For example, the molecular Hamiltonian for a water molecule can be constructed like this:
"""

import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "O", "H"]
geometry = np.array([[-0.0399, -0.0038, 0.0000],
                     [ 1.5780,  0.8540, 0.0000],
                     [ 2.7909, -0.5159, 0.0000]], requires_grad = False)

H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, method="pyscf")
print(H)

##############################################################################
# This generates a PennyLane :class:`~.pennylane.Hamiltonian` that can be used in a VQE workflow or
# converted to a
# `sparse matrix <https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html#sparse-hamiltonians>`_
# in the computational basis.
#
# Additionally, if you have built your electronic Hamiltonian independently using
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools, it can
# be readily converted to a PennyLane observable using the
# :func:`~.pennylane.import_operator` function. Here is an example:

from openfermion.ops import QubitOperator

H = 0.1 * QubitOperator('X0 X1') + 0.2 * QubitOperator('Z0')
H = qml.qchem.import_operator(H)

print(f'Type: \n {type(H)} \n')
print(f'Hamiltonian: \n {H}')

##############################################################################
# Computing molecular integrals
# -----------------------------
# In order to build a
# `molecular Hamiltonian <https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html>`_, we need
# one- and two-electron integrals in the molecular orbital basis. These integrals are used to
# construct a fermionic Hamiltonian which is then mapped onto the qubit basis. These molecular
# integrals can be computed with the
# :func:`~.pennylane.qchem.electron_integrals` function of PennyLane. Alternatively, the integrals
# can be computed with the `PySCF <https://github.com/sunqm/pyscf>`_ package and used in PennyLane
# workflows such as quantum resource estimation. Let's use water in the
# `6-31G basis <https://en.wikipedia.org/wiki/Basis_set_(chemistry)#Pople_basis_sets>`_ as
# an example.
#
# First, we define the PySCF molecule object and run a restricted Hartree-Fock
# calculation:

from pyscf import gto, ao2mo, scf

mol_pyscf = gto.M(atom = '''H -0.02111417 -0.00201087  0.;
                            O  0.83504162  0.45191733  0.;
                            H  1.47688065 -0.27300252  0.''', basis = '6-31g')
rhf = scf.RHF(mol_pyscf)
energy = rhf.kernel()

##############################################################################
# We obtain the molecular integrals ``one_ao`` and ``two_ao`` in the basis of atomic orbitals
# by following the example `here <https://pyscf.org/quickstart.html#and-2-electron-integrals>`_:

one_ao = mol_pyscf.intor_symmetric('int1e_kin') + mol_pyscf.intor_symmetric('int1e_nuc')
two_ao = mol_pyscf.intor('int2e_sph')

##############################################################################
# These integrals are then mapped to the basis of molecular orbitals:

one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)

##############################################################################
# Note that the two-electron integral tensor is represented in
# `chemists' notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_. To use it
# in PennyLane, we need to convert it into the so-called
# *physicists' notation*:

two_mo = np.swapaxes(two_mo, 1, 3)

##############################################################################
# Let's now look at an example where these molecular integrals are used to estimate the number of
# non-Clifford gates and logical qubits needed to implement a quantum phase estimation (QPE)
# algorithm. We use the computed integrals to estimate these resources for a
# `version of QPE <https://docs.pennylane.ai/en/stable/code/api/pennylane.resource.DoubleFactorization.html>`_
# that computes the expectation value of a double-factorized Hamiltonian in the second quantization.

algo = qml.resource.DoubleFactorization(one_mo, two_mo)

print(f'Estimated number of non-Clifford gates: {algo.gates:.2e}')
print(f'Estimated number of logical qubits: {algo.qubits}')

##############################################################################
# Conclusions
# -----------
# This tutorial demonstrates how to use PennyLane with external quantum chemistry libraries such as
# `PySCF <https://github.com/sunqm/pyscf>`_ and
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_.
#
# To summarize:
#
# 1. We can construct molecular Hamiltonians in PennyLane by using a user-installed version of PySCF by passing
#    the argument ``method=pyscf`` to the :func:`~.pennylane.qchem.molecular_hamiltonian` function.
# 2. We can directly use one- and two-electron integrals from PySCF, but we need to convert the
#    tensor containing the two-electron integrals from chemists' notation to physicists' notation.
# 3. Finally, we can easily convert OpenFermion operators to PennyLane operators using the 
#    :func:`~.pennylane.import_operator` function.
#
# About the author
# ----------------
# .. include:: ../_static/authors/soran_jahangiri.txt