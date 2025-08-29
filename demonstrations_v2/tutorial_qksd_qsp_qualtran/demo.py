r"""Using PennyLane and Qualtran to analyze how QSP can improve measurements of molecular properties
====================================================================================================

Want to efficiently measure molecular properties using quantum computers? This demo shows how to
use PennyLane to measure one- and two-particle reduced density matrices with a linearly-scaling
number of operations and how to integrate with Qualtran to demonstrate these resource requirements.
This is done by using Quantum Krylov Subspace Diagonalization (QKSD) techniques to "shrink" down a
complicated molecular Hamiltonian, find its ground-state classicaly, and then use Quantum Signal
Processing (QSP) to efficiently measure its one- and two-particle reduced density matrices. 

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-qualtran-covestro-krylov-subspace-paper-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In this demo we will demonstrate some of the techniques and results of the paper titled
`Molecular Properties from Quantum Krylov Subspace Diagonalization <https://arxiv.org/abs/2501.05286>`_
[#Oumarou]. Specifically, we will:

* Introduce QKSD briefly.
* Show how to build a PennyLane circuit that uses QSP to prepare the QKSD ground-state.
* Show how to simulate PennyLane circuits that estimate the one- and two-particle reduced density
    matrices of a molecular system from the QKSD ground-state.
* Show how to use the PennyLane-Qualtran integration to count relevant circuit resources
    and demonstrate linear resource scaling with respect to Krylov dimension, :math:`D`.
"""

######################################################################
# Quantum Krylov Subspace Diagonalization
# ---------------------------------------
#
# QKSD is a notable candidate algorithm for fault-tolerant quantum computing that estimates the
# eigenvalues and eigenstates of a large matrix by solving the eigenvalue problem for a smaller
# matrix [#QKSD]_. The general steps to perform QKSD are:
#
# * Obtain the Hamiltonian, :math:`\hat{H}`, describing a system of interest (e.g. a molecule).
# * Define a `Krylov subspace <https://en.wikipedia.org/wiki/Krylov_subspace>`_ :math:`\mathcal{K}`
#   of dimension :math:`D`, spanned by quantum states, :math:`| \psi_k \rangle`, that can be
#   efficiently prepared on a quantum computer.
# * Project :math:`\hat{H}` into the Krylov subspace with a quantum computer, resulting in a new,
#   "smaller" Hamiltonian :math:`\tilde{H}`. 
# * Calculate the overlap matrix, :math:`\tilde{S}`, defined by the inner-products of each element
#   of the Krylov subspace.
# * Solve the generalized eigenvalue problem:
#   :math:`\tilde{H}c^m = E_m \tilde{S}c^m` on a classical computer.
#
# The result of this generalized eigenvalue problem gives approximations of the low-lying
# eigenenergies and eigenstates of the Hamiltonian [#QKSD], including the Krylov ground-state,
# :math:`|\Psi_0\rangle = \sum_k c^0_k | \psi_k \rangle`.
#
# Such an eigenstate is a linear combination of the states spanning the Krylov subspace and can
# be prepared with `QSP <https://pennylane.ai/qml/demos/function_fitting_qsp>`_.
#
# Let's begin with the :math:`H_2O` molecule.
# We will use the Jordan-Wigner mapping of the
# :math:`H_2O` Hamiltonian with an active space of 4 electrons in 4
# molecular orbitals in the cc-pVDZ basis. To save time on this computationally-expensive basis, we
# provide pre-calculated coefficients and Pauli words for this Hamiltonian below. It is also possible
# to obtain these values using either `PennyLane Datasets <https://pennylane.ai/datasets/h2o-molecule>`_
# or the :mod:`pennylane.qchem` module. We use the precalculated results below to create a
# :class:`~pennylane.Hamiltonian` object.

import pennylane as qml
import numpy as np

coeffs = [-2.6055398817649027, 0.4725512342017669, 0.06908378485045799, 0.06908378485045799, 0.42465877221739423, 0.040785962774025165, 0.02016371425557293, 0.33650704944175736, 0.059002396986463035, 0.059002396986463035, 0.2401948108687125, 0.2696430914655511, 0.04971997220167805, 0.04971997220167805, 0.2751875205710399, 0.30033122257656397, 0.22551650339251875]
paulis = [qml.GlobalPhase(0, 0), qml.Z(0), qml.Y(0) @ qml.Z(1) @ qml.Y(2), qml.X(0) @ qml.Z(1) @ qml.X(2), qml.Z(1), qml.Z(2), qml.Z(3), qml.Z(0) @ qml.Z(1), qml.Y(0) @ qml.Y(2), qml.X(0) @ qml.X(2), qml.Z(0) @ qml.Z(2), qml.Z(0) @ qml.Z(3), qml.Y(0) @ qml.Z(1) @ qml.Y(2) @ qml.Z(3), qml.X(0) @ qml.Z(1) @ qml.X(2) @ qml.Z(3), qml.Z(1) @ qml.Z(2), qml.Z(1) @ qml.Z(3), qml.Z(2) @ qml.Z(3)]
hamiltonian = qml.Hamiltonian(coeffs, paulis)

######################################################################
# To define the Krylov subspace, we will use
# `Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials>`_ of the Hamiltonian
# applied to the Hartree-Fock state of the Hamiltonian, :math:`|\psi_0\rangle`. In other words, we
# define states: 
#
# .. math:: \ket{\psi_k} = T_i(H)\ket{\psi_0},
#
# where :math:`T_i` is the i-th Chebyshev polynomial. Then we define a Krylov subspace of dimension
# :math:`D` as 
#
# .. math:: \mathcal{K} = \text{span}(\{\ket{\psi_k}\}_{k=0}^{D-1}).
#
# With the Hamiltonian and Krylov space defined, we can use
# the `Lanczos method <https://quantum-journal.org/papers/q-2023-05-23-1018/pdf/>`_ classically
# or `QKSD <https://arxiv.org/pdf/2407.14431>`_ to find :math:`\tilde{H}`
# and :math:`\tilde{S}`, then solve for the QKSD ground-state,
# 
# .. math:: |\Psi_0\rangle = \sum_k c^0_k | \psi_k \rangle = \sum_{i=0}^{D-1}c_iT_i(H)\ket{\psi_0},
#
# where :math:`c_i` are the cofficients of the :math:`i`-th Chebyshev polynomial.
# Here we use the Chebyshev basis because QSP directly implements Chebyshev polynomials.
# Other types of functions need to be converted into Chebyshev polynomials to implement via QSP.
#
# Using QSP to directly create the QKSD ground-state
# --------------------------------------------------
#
# Only using QKSD to calculate one- and two-particle reduced density matrices
# results in a quadratic scaling of the number of expectation
# values that need to be measured with respect to the Krylov dimension, :math:`D` [#Oumarou].
# Instead, reference [#Oumarou] shows that it is possible to reduce this to a constant scaling by preparing
# the QKSD ground-state, :math:`|\Psi_0\rangle`, via QSP and measuring individual terms of the Hamiltonian.
# [TODO: clarify QSP explanation below. Too many unclear variables and not obvious how qsp implements a polynomial+no block encodings mentioned]
# The QSP circuit, :math:`U_{qsp}`, is defined as a series of alternating iterates,
# :math:`W`, and rotations, :math:`S`, such that:
# 
# .. math:: U_\text{qsp} = S(\phi_0)\prod_k^{d-1}{W(a)S(\phi_k)}
# 
# Choosing the right rotation angles, :math:`\phi`, causes :math:`U_\text{qsp}` to implement a 
# Chebyshev polynomial such that:
#
# .. math:: U_{qsp}|\psi_0\rangle|\psi_a\rangle = \sum_{i=0}^{D-1}c_iT_i(H)\ket{\psi_0} = \Psi_0\rangle|\psi_a'\rangle
#
# This will prepare the QKSD ground-state by implementing the corresponding Chebyshev polynomial of
# the block-encoded Hamiltonian. For more details, see
# `Function Fitting using Quantum Signal Processing <https://pennylane.ai/qml/demos/function_fitting_qsp>`_.
# Let's see how to do this in PennyLane.
#
# Defining the QSP circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# To define the QSP circuit, we first create a function to implement the rotations, :math:`S`. The
# iterate, :math:`W`, will be implemented via the :class:`~pennylane.PrepSelPrep` class. We then
# create a QSP template to alternate between :math:`W` and :math:`S`. Alternating between these operations
# will prepare a Chebyshev polynomial of the input block-encoded matrix.

def rotation_about_reflection_axis(angle, wires):
    qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))
    qml.RZ(angle, wires[0])
    qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))

def qsp(lcu, angles, rot_wires, prep_wires):
    for angle in angles[::-1][:-1]:
        rotation_about_reflection_axis(angle, rot_wires)
        qml.PrepSelPrep(lcu, control=prep_wires)
    rotation_about_reflection_axis(angles[0], rot_wires)

######################################################################
# Since QSP can only produce fixed-parity real Chebyshev polynomials [#QSP] and our QKSD ground-state has
# complex polynomials, we also create a template that combines the real an imaginary parts of the
# polynomial.

def qsp_poly_complex(lcu, angles_real, angles_imag, ctrl_wire, rot_wires, prep_wires):
    qml.H(ctrl_wire)
    qml.ctrl(qsp, ctrl_wire, 0)(lcu, angles_real, rot_wires, prep_wires)
    qml.ctrl(qsp, ctrl_wire, 1)(lcu, angles_imag, rot_wires, prep_wires)
    qml.H(ctrl_wire)

######################################################################
# Additionally, we require a template to perform a reflection for our measurements.
# [TODO: explain measurements and restructure this section]
#
def reflection(wire, ctrl_wires):
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))
    qml.Z(wire)
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))

######################################################################
# We will also need to split even- and odd-parity polynomial terms. But this is captured in the
# final circuit below. Since we have split the polynomial and implement each part via a separate QSP
# circuit, we will also need to split the rotation angles that implement each of these polynomials.
# The final circuit below accepts a separate argument for odd/even and real/imaginary angles, these
# are names ``even_real``, ``even_imag``, ``odd_real``, and ``odd_imag``.
#
# We also define an argument for an input observable, ``obs``, and an argument defining whether to
# measure the reflection of the observable, ``measure_reflection``. 
#
# With these building blocks in place, we can now define the overarching QNode
# that will implement the QSP circuit and return measurements of the elements of the one-particle
# and two-particle reduced density matrices:

dev = qml.device("lightning.qubit")

@qml.qnode(dev)
def krylov_qsp(lcu, even_real, even_imag, odd_real, odd_imag, obs, measure_reflection=False):
    """Prepares the Krylov lowest-energy state by applying QSP with the input angles.
    Then measures the expectation value of the desired observable. 
    """
    num_aux = int(np.log(len(lcu.operands)) / np.log(2)) + 1

    start_wire = hamiltonian.wires[-1] + 1
    rot_wires = list(range(start_wire, start_wire + num_aux + 1))
    prep_wires = rot_wires[1:]
    ctrl_wires = [prep_wires[-1] + 1, prep_wires[-1] + 2]
    rdm_ctrl_wire = ctrl_wires[-1] + 1

    if measure_reflection: # preprocessing for reflection measurement
        qml.X(rdm_ctrl_wire)

    #[TODO: explain why we are combining two QSP calls again here]
    qml.H(ctrl_wires[0])
    qml.ctrl(qsp_poly_complex, ctrl_wires[0], 0)(lcu, even_real, even_imag, ctrl_wires[1], rot_wires, prep_wires)
    qml.ctrl(qsp_poly_complex, ctrl_wires[0], 1)(lcu, odd_real, odd_imag, ctrl_wires[1], rot_wires, prep_wires)
    qml.H(ctrl_wires[0])

    # measurements
    if measure_reflection:
        return qml.expval(qml.prod(reflection)(rdm_ctrl_wire, set(ctrl_wires+prep_wires))@obs)
    return qml.expval(obs)


######################################################################
# We can now use this circuit to build one- and two- particle reduced density matrices.
# Based on reference [#Oumarou],
# the elements of the one-particle reduced density matrix are obtained by measuring the expectation
# value of the fermionic one-particle excitation operators acting on the Krylov lowest energy state:
#
# .. math:: \langle\Psi_0 | \hat{E}_{pq} | \Psi_0\rangle.
#
# We use the Jordan-Wigner mapping of the fermionic one-particle excitation operators and measure
# the resulting Pauli word observables.
# [TODO: demonstrate what the output value of the coherent result is, put it into context]
# We can obtain the Jordan-Wigner mapping of the fermionic operators via PennyLane using the
# :func:`~pennylane.fermi.from_string` and :func:`~pennylane.jordan_wigner` functions as follows:

Epq = qml.fermi.from_string('0+ 0-')
obs = qml.jordan_wigner(Epq)

######################################################################
# For this demo, we used a Krylov subspace dimension of :math:`D=15` and pre-computed QSP angles
# that implement the corresponding sum of Chebyshev polynomials :math:`\sum_{i=0}^{D-1}c_iT_i`.
# For a given polynomial it is possible to obtain the QSP angles using :func:`~pennylane.poly_to_angles`.
#
# The angles below will produce the QKSD ground-state :math:`|\Psi_0\rangle` via QSP. Since QSP
# can only produce fixed-parity real Chebyshev polynomials [#QSP] and our ground-state QKSD has
# mixed-parity complex polynomials, we split them and apply separately.

even_real = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
even_imag = np.array([3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173])
odd_real = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
odd_imag = np.array([3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691])
######################################################################
# We then measure these and post-process according to Equation 32 of the paper:
# 
# .. math:: 2\langle \Psi_0 |_s\hat{P}_{\nu}|\Psi_0\rangle_s = \eta^2(o_1 + o_2).
#

measurement_1 = krylov_qsp(hamiltonian, even_real, even_imag, odd_real, odd_imag, obs=obs)
measurement_2 = krylov_qsp(hamiltonian, even_real, even_imag, odd_real, odd_imag, obs=obs, measure_reflection=True)

print("meas 1:", measurement_1)
print("meas 2:", measurement_2)

lambda_lcu = np.sum(np.abs(coeffs))
coherent_result = 2*lambda_lcu*(measurement_1+measurement_2)
print("coherent result:",coherent_result)

######################################################################
# Analyzing with Qualtran
# -----------------------
# 
# We can analyze the resources and flow of this program by using the Qualtran call graph.
# We first convert the PennyLane circuit to a Qualtran bloq and then use the call graph to count
# the required gates:

bloq = qml.to_bloq(krylov_qsp, map_ops=False,
    even_real=even_real, even_imag=even_imag,
    odd_real=odd_real, odd_imag=odd_imag,
    lcu=hamiltonian, obs=obs
    )

######################################################################
# We can then use Qualtran tools to analyze and process the gate counts of the circuit. For example,
# we can use the ``call_graph`` to obtain a breakdown of the gates used:


graph, sigma = bloq.call_graph()
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# We can also apply the ``generalize_rotation_angle`` generalizer to neatly group all rotations for
# clearer viewing:

from qualtran.resource_counting.generalizers import generalize_rotation_angle

graph, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# As explained in [#Oumarou],
# increasing the dimension, :math:`D`, of the Krylov subspace improves the accuracy of the Krylov minimal energy
# compared to the true ground state energy. This extra accuracy is paid for by requiring additional gates.
# Let's see how the number of gates increases with increasing Krylov susbspace dimension. We can
# increase the Krylov subspace dimension by increasing the number of terms in our Chebyshev polynomial,
# captured in this demo via the angles variables. Let's try :math:`D=20`
# by setting the number of terms in these angles to 20. As the resource estimation is independent of
# the exact angle values, we are able to set them randomly instead of recomputing the formally:

even_real = even_imag = odd_real = odd_imag = np.random.random(20)

######################################################################
# We then repeat the gate counts and see they have increased:

bloq = qml.to_bloq(krylov_qsp, map_ops=False,
    even_real=even_real, even_imag=even_imag,
    odd_real=odd_real, odd_imag=odd_imag,
    lcu=hamiltonian, obs=obs
    )
graph, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# We can plot how the number of gates increases with the Krylov dimension to see if it is linear
# as described in [#Oumarou]. Below we plot how the Toffoli, CNOT, and X gate count increase with the Krylov dimension:

import matplotlib.pyplot as plt
from qualtran.bloqs.basic_gates import Toffoli, CNOT, XGate

def count_cnots(krylov_dimension):
    even_real = even_imag = odd_real = odd_imag = np.random.random(krylov_dimension)
    bloq = qml.to_bloq(krylov_qsp, map_ops=False,
                   even_real=even_real, even_imag=even_imag,
                    odd_real=odd_real, odd_imag=odd_imag,
                     lcu=hamiltonian, obs=obs)
    _, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
    return sigma

Ds = [10, 20, 30, 40, 50]
sigmas = [count_cnots(D) for D in Ds]

plt.style.use("pennylane.drawer.plot")
plt.scatter(Ds, [sigma[CNOT()] for sigma in sigmas], label='CNOTs')
plt.scatter(Ds, [sigma[Toffoli()] for sigma in sigmas], label = 'Toffolis')
plt.scatter(Ds, [sigma[XGate()] for sigma in sigmas], label = 'X Gates')
plt.xlabel('Krylov Subspace Dimension, D')
plt.ylabel('Number of Gates')
plt.legend()

######################################################################
# As expected, the gate counts increase linearly with the Krylov subspace dimension.
# 
# We can also show the call graph of the circuit, a diagrammatic representation of how each operation
# in the circuit calls other operations. This can be useful to understand how a circuit works or 
# which sections are resource-intensive.
#
# The first layer of this circuit, before decomposing the operations, contains many
# ``RZ`` gates controlled on two qubits with different angles. As we saw above, we can group similar
# operations together and simplify the view. To this end, we first write a custom Qualtran generalizer
# that will combine controlled-``RZ`` gates: 

import attrs
import sympy

from qualtran.resource_counting.generalizers import _ignore_wrapper

PHI = sympy.Symbol(r'\phi')

def generalize_ccrz(b):
    from qualtran.bloqs.basic_gates import Rz
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    if isinstance(b, ControlledViaAnd):
        return attrs.evolve(b, subbloq = Rz(angle=PHI))
    
    return _ignore_wrapper(generalize_ccrz, b)

######################################################################
# We can then use this generalizer to draw our call graph using Qualtran's ``show_call_graph``
# [TODO: replace this code block with a non-executable one and manually add the static image output]

from qualtran.drawing import show_call_graph

graph, sigma = bloq.call_graph(generalizer=generalize_ccrz)
show_call_graph(graph)

######################################################################
# Conclusion
# ----------
# [TODO]
#
# References
# -----------
#
# .. [#QSP]
#
#     Guang Hao Low, Isaac L. Chuang
#     "Hamiltonian Simulation by Qubitization",
#     `arXiv:1610.06546 <https://arxiv.org/abs/1610.06546v3>`__, 2019.
# 
# .. [#Oumarou]
#
#     Oumarou Oumarou, Pauline J. Ollitrault, Cristian L. Cortes, Maximilian Scheurer, Robert M. Parrish, Christian Gogolin
#     "Molecular Properties from Quantum Krylov Subspace Diagonalization",
#     `arXiv:2501.05286 <https://arxiv.org/abs/2501.05286>`__, 2025.
#
# .. [#QKSD]
#
#     Nobuyuki Yoshioka, Mirko Amico, William Kirby, Petar Jurcevic, Arkopal Dutt, Bryce Fuller, Shelly Garion,
#     Holger Haas, Ikko Hamamura, Alexander Ivrii, Ritajit Majumdar, Zlatko Minev, Mario Motta, Bibek Pokharel,
#     Pedro Rivero, Kunal Sharma, Christopher J. Wood, Ali Javadi-Abhari, and Antonio Mezzacapo
#     "Diagonalization of large many-body Hamiltonians on a quantum processor",
#     `arXiv:2407.14431 <https://arxiv.org/abs/2407.14431>`__, 2024.
#
# About the author
# ----------------
#
