r"""Using PennyLane and Qualtran to analyze how QSP can improve measurements of molecular properties
======================================

Want to efficiently measure molecular properties using quantum computers? After simplifying a 
molecule using Quantum Krylov Subspace Diagonalization (QKSD) techniques, Quantum Signal Processing (QSP)
can be used to efficiently measure the one- and two-particle reduced density matrices of molecular
systems. 

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-qualtran-qksd.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

In this demo we'll follow
[Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286)
to:

* Briefly introduce QKSD.
* Estimate the reduced density matrices, :math:`\gamma_{pq}` and :math:`\Gamma_{pqrs}` of a polynomial
    of the Hamiltonian applied to the QKSD lowest-energy state.
* Use the PennyLane-Qualtran integration to count the number of qubits and gates required by these
    circuits and demonstrate the scaling with respect to Krylov dimension, :math:`D`.

"""

######################################################################
# Quantum Krylov Subspace Diagonalization
# ---------------------------------------
#
# The exact details of Quantum Krylov Subspace Diagonalization (QKSD) are beyond the scope of
# this demo. However, the general outline of the technique is as follows:
#
# * Choose a molecule and obtain its Hamiltonian (:math:`\hat{H}`)
# * Define a Krylov subspace for the desired molecule.
# * Calculate the projection of the Hamiltonian into the subspace (:math:`\tilde{H}`) 
#       and the overlap matrix (:math:`\tilde{S}`)
# * Solve the generalized eigenvalue problem: :math:`\tiled{H}c^m = E_m \tilde{S}c^m`
#
# Solving this generalized eigenvalue problem gives approximations of the possible energies of
# the molecule, like the lowest Krylov energy :math:`|\Psi_0\rangle`
# 
# For the purposes of this demo, we will use a set of pre-calculated QSP angles that implement the
# lowest-energy Krylov eigenstate in the Chebyshev basis.
#  
# Let's begin with the :math:`H_2O` molecule. We can obtain the details of
# this molecule using PennyLane's datasets as follows:

import pennylane as qml
import numpy as np

[ds] = qml.data.load("qchem", molname="H2O", bondlength=0.958, basis="STO-3G", attributes=["molecule"])
molecule = ds.molecule

######################################################################
# We can then calculate the one-electron and two-electron terms of the hamiltonian using the ``qchem``
# module: 

const, h1, h2 = qml.qchem.electron_integrals(molecule, core=[0, 1, 2], active=[3, 4, 5, 6])()
hpq, hpqrs = np.array(h1), np.array(h2)

######################################################################
# Based on these values we can generate the Hamiltonian of the system:

from openfermion.ops import InteractionOperator
from openfermion.transforms import get_fermion_operator

interaction_op = InteractionOperator(constant=0.0, one_body_tensor=hpq, two_body_tensor=hpqrs)
fermion_hamiltonian = get_fermion_operator(interaction_op)
hamiltonian = qml.qchem.observable(fermion_hamiltonian)
coeffs, paulis = hamiltonian.terms()

######################################################################
# Using QSP to directly create the QKSD ground-state
# --------------------------------------------------
#
# With the Hamiltonian defined, we proceed to implement the QKSD ground-state :math:`|\Psi_0\rangle`
# via QSP.
# In this case, we pre-computed the required values for the
# :math:`H_2O` molecule using a Krylov subspace of dimension :math:`D=15`.
# These values can be obtained using e.g. the [lanczos method](https://quantum-journal.org/papers/q-2023-05-23-1018/pdf/)
# or using [Quantum Krylov Subspace diagonalization](https://arxiv.org/pdf/2407.14431) to find the 
# QKSD ground-state, :math:`|\Psi_0\rangle`, in the Chebyshev basis defined by:
# 
# .. math:: \ket{\psi_k} \coloneqq \sum_{i=0}^{D-1}c^k_iT_i(H)\ket{\psi_0},
# 
# .. math:: \mathcal{K}& \coloneqq \lspan(\{\ket{\psi_k}\}_{k=0}^{D-1}),
# 
# The angles below will be used to produce the QKSD ground-state :math:`|\Psi_0\rangle` via QSP, as
# defined by:
# 
# .. math:: \ket{\Psi_0}=\sum_{i=0}^{D-1} c^0_i T_i(\hat{H})\ket{\psi_0}

angles_even_real = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
angles_even_imag = np.array([3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173])
angles_odd_real = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
angles_odd_imag = np.array([3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691])

######################################################################
# Defining the QSP circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~
# We can now create operations to implement the QSP circuit that will prepare the QKSD ground-state.
# For this we need to create a rotation operator or signal processing operator, and a QSP template
# to alternate between this and a block-encoding operator. For more details, see
# [Function Fitting using Quantum Signal Processing](https://pennylane.ai/qml/demos/function_fitting_qsp)

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
# We also add a template to combine real and imaginary parts of the Chebyshev polynomials as QSP
# can only implement one at a time:

def qsp_poly_complex(lcu, angles_real, angles_imag, ctrl_wire, rot_wires, prep_wires):
    qml.H(ctrl_wire)
    qml.ctrl(qsp, ctrl_wire, 0)(lcu, angles_real, rot_wires, prep_wires)
    qml.ctrl(qsp, ctrl_wire, 1)(lcu, angles_imag, rot_wires, prep_wires)
    qml.H(ctrl_wire)

######################################################################
# And a template to perform a reflection for our measurements.
# [TODO: explain measurements and restructure this section]
#
def reflection(wire, ctrl_wires):
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))
    qml.Z(wire)
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))

######################################################################
#
# With these building blocks in place, we can then define the overarching QNode
# that will implement this circuit:

dev = qml.device("lightning.qubit")

@qml.qnode(dev)
def krylov_qsp(lcu, angles_even_real, angles_even_imag, angles_odd_real, angles_odd_imag, measure=0, obs=None):

    num_ancillae = int(np.log(len(lcu.operands)) / np.log(2)) + 1

    start_wire = hamiltonian.wires[-1] + 1
    rot_wires = list(range(start_wire, start_wire + num_ancillae + 1))
    prep_wires = rot_wires[1:]
    ctrl_wires = [prep_wires[-1] + 1, prep_wires[-1] + 2]
    rdm_ctrl_wire = ctrl_wires[-1] + 1
    
    if measure==2: # preprocessing for reflection measurement
        qml.X(rdm_ctrl_wire)
    
    qml.H(ctrl_wires[0])
    qml.ctrl(qsp_poly_complex, ctrl_wires[0], 0)(lcu, angles_even_real, angles_even_imag, ctrl_wires[1], rot_wires, prep_wires)
    qml.ctrl(qsp_poly_complex, ctrl_wires[0], 1)(lcu, angles_odd_real, angles_odd_imag, ctrl_wires[1], rot_wires, prep_wires)
    qml.H(ctrl_wires[0])

    # measurements
    if measure==1:
        return qml.expval(obs)
    if measure==2:
        return qml.expval(
            qml.prod(reflection)(rdm_ctrl_wire, set(ctrl_wires+prep_wires))
            @
            obs
        )
    return qml.state()


######################################################################
# We can use these to build 1- and 2- particle reduced density matrices
# Based on [Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286)
# [TODO: repeat measurements for different Pauli Words and build reduced density matrices]
# [TODO: demonstrate reduced density matrices are correct]

obs = qml.H(hamiltonian.wires[-1]) @ qml.H(hamiltonian.wires[-2])

measurement_1 = krylov_qsp(hamiltonian, angles_even_real, angles_even_imag, angles_odd_real, angles_odd_imag, measure=1, obs=obs)
measurement_2 = krylov_qsp(hamiltonian, angles_even_real, angles_even_imag, angles_odd_real, angles_odd_imag, measure=2, obs=obs)

print("meas 1:", measurement_1)
print("meas 2:", measurement_2)

lambda_lcu = np.sum(np.abs(coeffs))
coherent_result = 2*lambda_lcu*(measurement_1+measurement_2)
print("coherent result:",coherent_result)

######################################################################
#
# We can analyze the resources and flow of this program by using the qualtran call graph.
# We first convert the PennyLane circuit to a Qualtran bloq and then use the call graph to count
# the required gates:

hamiltonian = qml.Hamiltonian(hamiltonian.terms()[0], [qml.GlobalPhase(0,wires=hamiltonian.wires[0]) if i.name=="Identity" else i for i in hamiltonian.terms()[1]])
# ^ redefining Identity as Global Phase in hamiltonian as a temprorary work-around
bloq = qml.to_bloq(krylov_qsp, map_ops=False,
                   angles_even_real=angles_even_real, angles_even_imag=angles_even_imag,
                    angles_odd_real=angles_odd_real, angles_odd_imag=angles_odd_imag,
                     lcu=hamiltonian, measure=1, obs=obs)

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
# As explained in
# [Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286),
# increasing the dimension of the Krylov subspace improves the accuracy of the Krylov minimal energy
# compared to the true ground state energy. This extra accuracy is paid by requiring additional gates.
# Let's see how the number of gates increases with increasing Krylov susbspace dimension. We can
# increase the Krylov subspace dimension by increasing the number of terms in our Chebyshev polynomial,
# captured in this demo via the angles variables. Let's try increasing the Krylov dimension to 20
# by setting the number of terms in these angles to 20. We can choose random angles as the resource
# estimation is independent of the exact angle values:

angles_even_real = angles_even_imag = angles_odd_real = angles_odd_imag = np.random.random(20)

######################################################################
# We then repeat the gate counts and see they have increased:

bloq = qml.to_bloq(krylov_qsp, map_ops=False,
                   angles_even_real=angles_even_real, angles_even_imag=angles_even_imag,
                    angles_odd_real=angles_odd_real, angles_odd_imag=angles_odd_imag,
                     lcu=hamiltonian, measure=1, obs=obs)
graph, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# We can plot how e.g. the number of CNOT gates increases with the Krylov dimension as follows:

import matplotlib.pyplot as plt
from qualtran.bloqs.basic_gates import CNOT

def count_cnots(krylov_dimension):
    angles_even_real = angles_even_imag = angles_odd_real = angles_odd_imag = np.random.random(krylov_dimension)
    bloq = qml.to_bloq(krylov_qsp, map_ops=False,
                   angles_even_real=angles_even_real, angles_even_imag=angles_even_imag,
                    angles_odd_real=angles_odd_real, angles_odd_imag=angles_odd_imag,
                     lcu=hamiltonian, measure=1, obs=obs)
    _, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
    return sigma[CNOT()]

Ds = [10, 20, 30, 40, 50]
cnots = [count_cnots(D) for D in Ds]
plt.plot(Ds, cnots)

######################################################################
# Conclusion
# ----------
# [TODO]
#
# About the author
# ----------------
#
