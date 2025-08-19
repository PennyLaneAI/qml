r"""demonstrations_v2/tutorial_qksd_qualtran/metadata.json
======================================

Want to shrink large molecules down to a more manageable size? Quantum Krylov subspace
diagonalization techniques do just that, by mapping the molecular Hamiltonian down to a smaller
Krylov subspace. This mapping can then be used to obtain molecular properties such as the system's
reduced density matrices and nuclear gradients of the energy. [are these two properties and their usefulness
clear enough to researchers or do we need to rephrase?]

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-qualtran-qksd.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In this demo we'll follow [Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286)
to explain how to:

* Estimate the reduced density matrices of a polynomial of the Hamiltonian applied to a given state
    [not immediately clear what a polynomial of the Hamiltonian applied to a given state is useful for]
* Use the PennyLane-Qualtran integration to count the number of qubits and gates required by these
    circuits.

"""

######################################################################
# Estimating reduced density matrices
# -----------------------------------
#
# The first steps in Quantum Krylov methods are to:
#
# * Choose a molecule and obtain its Hamiltonian in the Jordan Wigner mapping
# * Define a Krylov subspace for the desired molecule.
# * Classically calculate the projection of the Hamiltonian into the subspace (:math:`\tilde{H}`) 
#       and the overlap matrix (:math:`\tilde{S}`) 
#
# In this case, we choose the :math:`H_2O` molecule. We can obtain the Jordan-Wigner Hamiltonian using 
# PennyLane as follows:

import pennylane as qml
import numpy as np

symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0], [1.5780, 0.8540, 0.0], [2.7909, -0.5159, 0.0]])

molecule = qml.qchem.Molecule(symbols, coordinates)
H, qubits = qml.qchem.molecular_hamiltonian(molecule)
print(H)

######################################################################
# Many subspaces are possible, but in this case we choose the span of the first 15 Chebyshev
# polynomials of the Hamiltonian applied to the Hartree-Fock ground state: 
#
#  .. math:: |\psi_k\rangle = U_{qsp(k)}|\psi_{hf}\rangle
#
# Using a Krylov subspace of dimension :math:`D=15`, we pre-computed the required values [what values?] for the
# :math:`H_2O` molecule. This can be done using e.g. the [lanczos method](https://quantum-journal.org/papers/q-2023-05-23-1018/pdf/)
# [is this correct?]
# 
# We project the Hamiltonian of the 
# 
# We provide the precalculated values for an :math:`H_2O` molecule with Krylov dimension 15:

poly_coeffs = np.array([0.04839504, -0.11122954, -0.11047445, 0.19321127, 0.10155112, -0.07727596, 0.12920728, 0.03954071, -0.25404897, -0.03059887, -0.00703527, -0.28554963, 0.14565572, -0.1226918, 0.02780524])
angles_even = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
angles_conj_even = np.array([[3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173]])
angles_odd = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
angles_conj_odd = [3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691]
poly_degree = len(poly_coeffs) - 1

######################################################################
# We can use these to build
# Based on [Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286)

import pennylane as qml

# do some stuff

######################################################################
# Explanation about code

import numpy as np
# additional code

######################################################################
# Explanation about code
#
# potential subsection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Subsection explanation and preparation for code

from qualtran.bloqs import something

######################################################################


######################################################################
# 
#
# Second Section
# ------------------------------------------
#
# 

######################################################################
# Conclusion
# ----------
# In this demo, we did something.
#
# About the author
# ----------------
#
