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
# Introduction to the method and initial steps to set it up
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
