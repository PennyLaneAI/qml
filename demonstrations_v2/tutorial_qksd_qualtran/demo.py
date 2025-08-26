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

In this demo we'll follow
[Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286)
to explain how to:

* Estimate the reduced density matrices of a polynomial of the Hamiltonian applied to a given state
* Use the PennyLane-Qualtran integration to count the number of qubits and gates required by these
    circuits.

"""

######################################################################
# Quantum Krylov Subspace Diagonalization
# ---------------------------------------
#
# The first steps in Quantum Krylov methods are to:
#
# * Choose a molecule and obtain its Hamiltonian in the Jordan Wigner mapping (:math:`\hat{H}`)
# * Define a Krylov subspace for the desired molecule.
# * Classically calculate the projection of the Hamiltonian into the subspace (:math:`\tilde{H}`) 
#       and the overlap matrix (:math:`\tilde{S}`) 
#
# In this case, we choose the :math:`H_2O` molecule. We can obtain the Jordan-Wigner Hamiltonian, :math:`\hat{H}`,
# using PennyLane as follows:

import pennylane as qml
import numpy as np

symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0], [1.5780, 0.8540, 0.0], [2.7909, -0.5159, 0.0]])

molecule = qml.qchem.Molecule(symbols, coordinates)
H, qubits = qml.qchem.molecular_hamiltonian(symbols,coordinates)

const, h1, h2 = qml.qchem.electron_integrals(molecule, core=[0, 1, 2], active=[3, 4, 5, 6])()
hpq, hpqrs = np.array(h1), np.array(h2)

######################################################################
# With the Hamiltonian defined, we proceed to calculate the Krylov subspace in the Chebyshev basis. 
# Many subspaces are possible, but in this case we pre-computed the required values [what values?] for the
# :math:`H_2O` molecule using a Krylov subspace of dimension :math:`D=15`.
 
hpq = np.array([[-2.50986452e00, 1.05657350e-16, -2.22121281e-01, -2.86068006e-02],[3.16667563e-16, -2.51194446e00, 1.21911191e-15, -5.06049410e-16],[-2.22121281e-01, 1.17886295e-15, -1.21501454e00, -2.54198372e-02],[-2.86068006e-02, -6.44098020e-16, -2.54198372e-02, -1.10480366e00],])
hpqrs = np.array([[[[6.96056671e-01, 8.64123979e-17, 7.51963671e-02, 9.51662480e-03],[1.71207496e-16, 4.07411355e-02, -1.45820198e-17, -3.87762611e-18],[7.51963671e-02, -1.67452401e-17, 3.05202191e-02, 5.34230841e-03],[9.51662480e-03, 2.61368866e-17, 5.34230841e-03, 1.67384743e-02],],[[1.71207496e-16, 4.07411355e-02, -1.45820198e-17, -3.87762611e-18],[6.50690725e-01, 1.23045003e-17, 7.46798397e-02, 1.01652275e-02],[-2.47658344e-16, 2.43546325e-03, -4.51712874e-17, 2.48872553e-17],[2.25988600e-16, 1.23962208e-03, 6.53416581e-17, -4.61937088e-17],],[[7.51963671e-02, -1.67452401e-17, 3.05202191e-02, 5.34230841e-03],[-2.47658344e-16, 2.43546325e-03, -4.51712874e-17, 2.48872553e-17],[3.54579407e-01, -3.60619099e-17, 5.55938827e-03, -2.76946416e-03],[1.02943160e-02, 2.41224973e-17, -5.75780811e-03, -2.10651109e-02],],[[9.51662480e-03, 2.61368866e-17, 5.34230841e-03, 1.67384743e-02],[2.25988600e-16, 1.23962208e-03, 6.53416581e-17, -4.61937088e-17],[1.02943160e-02, 2.41224973e-17, -5.75780811e-03, -2.10651109e-02],[3.41425718e-01, -2.31491583e-17, 3.85389582e-03, 7.06935786e-03],],],[[[8.64123979e-17, 6.50690725e-01, -2.78377815e-16, 2.49516124e-16],[4.07411355e-02, 6.27787581e-18, 2.43546325e-03, 1.23962208e-03],[-1.67452401e-17, 7.46798397e-02, -5.60235836e-17, 6.70915608e-17],[2.61368866e-17, 1.01652275e-02, 2.78665353e-17, -5.04507066e-17],],[[4.07411355e-02, 6.27787581e-18, 2.43546325e-03, 1.23962208e-03],[1.23045003e-17, 7.62241410e-01, -3.05861696e-16, 3.55270558e-16],[2.43546325e-03, -3.26993657e-16, 1.24617768e-02, -3.33498976e-06],[1.23962208e-03, 3.18178705e-16, -3.33498976e-06, 7.25183601e-03],],[[-1.67452401e-17, 7.46798397e-02, -5.60235836e-17, 6.70915608e-17],[2.43546325e-03, -3.26993657e-16, 1.24617768e-02, -3.33498976e-06],[-3.60619099e-17, 3.59082675e-01, -5.69105903e-17, 9.60811644e-18],[2.41224973e-17, 5.08519862e-03, -9.99415338e-17, 1.14310026e-16],],[[2.61368866e-17, 1.01652275e-02, 2.78665353e-17, -5.04507066e-17],[1.23962208e-03, 3.18178705e-16, -3.33498976e-06, 7.25183601e-03],[2.41224973e-17, 5.08519862e-03, -9.99415338e-17, 1.14310026e-16],[-2.31491583e-17, 3.44431171e-01, -2.81970002e-17, -2.33497921e-17],],],[[[7.51963671e-02, -2.78377815e-16, 3.54579407e-01, 1.02943160e-02],[-1.45820198e-17, 2.43546325e-03, 5.56527542e-17, 6.35026535e-17],[3.05202191e-02, -5.60235836e-17, 5.55938827e-03, -5.75780811e-03],[5.34230841e-03, 2.78665353e-17, -2.76946416e-03, -2.10651109e-02],],[[-1.45820198e-17, 2.43546325e-03, 5.56527542e-17, 6.35026535e-17],[7.46798397e-02, -3.05861696e-16, 3.59082675e-01, 5.08519862e-03],[-4.51712874e-17, 1.24617768e-02, -5.24368327e-17, -1.09398900e-16],[6.53416581e-17, -3.33498976e-06, 3.62832929e-17, 9.42506062e-17],],[[3.05202191e-02, -5.60235836e-17, 5.55938827e-03, -5.75780811e-03],[-4.51712874e-17, 1.24617768e-02, -5.24368327e-17, -1.09398900e-16],[5.55938827e-03, -5.69105903e-17, 3.12395481e-01, 1.60872665e-02],[-5.75780811e-03, -9.99415338e-17, 1.60872665e-02, 6.97375883e-02],],[[5.34230841e-03, 2.78665353e-17, -2.76946416e-03, -2.10651109e-02],[6.53416581e-17, -3.33498976e-06, 3.62832929e-17, 9.42506062e-17],[-5.75780811e-03, -9.99415338e-17, 1.60872665e-02, 6.97375883e-02],[3.85389582e-03, -2.81970002e-17, 2.94109098e-01, -1.19378613e-02],],],[[[9.51662480e-03, 2.49516124e-16, 1.02943160e-02, 3.41425718e-01],[-3.87762611e-18, 1.23962208e-03, 6.35026535e-17, -4.14362821e-17],[5.34230841e-03, 6.70915608e-17, -5.75780811e-03, 3.85389582e-03],[1.67384743e-02, -5.04507066e-17, -2.10651109e-02, 7.06935786e-03],],[[-3.87762611e-18, 1.23962208e-03, 6.35026535e-17, -4.14362821e-17],[1.01652275e-02, 3.55270558e-16, 5.08519862e-03, 3.44431171e-01],[2.48872553e-17, -3.33498976e-06, -1.09398900e-16, -2.46748276e-18],[-4.61937088e-17, 7.25183601e-03, 9.42506062e-17, -4.91354497e-18],],[[5.34230841e-03, 6.70915608e-17, -5.75780811e-03, 3.85389582e-03],[2.48872553e-17, -3.33498976e-06, -1.09398900e-16, -2.46748276e-18],[-2.76946416e-03, 9.60811644e-18, 1.60872665e-02, 2.94109098e-01],[-2.10651109e-02, 1.14310026e-16, 6.97375883e-02, -1.19378613e-02],],[[1.67384743e-02, -5.04507066e-17, -2.10651109e-02, 7.06935786e-03],[-4.61937088e-17, 7.25183601e-03, 9.42506062e-17, -4.91354497e-18],[-2.10651109e-02, 1.14310026e-16, 6.97375883e-02, -1.19378613e-02],[7.06935786e-03, -2.33497921e-17, -1.19378613e-02, 3.06238461e-01],],],])

angles_even_real = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
angles_even_imag = np.array([3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173])
angles_odd_real = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
angles_odd_imag = np.array([3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691])

######################################################################
# These values can be obtained using e.g. the [lanczos method](https://quantum-journal.org/papers/q-2023-05-23-1018/pdf/)
# [is this correct?] or using [Quantum Krylov Subspace diagonalization](https://arxiv.org/pdf/2407.14431).


hpq = np.array([[-2.50986452e00, 1.05657350e-16, -2.22121281e-01, -2.86068006e-02],[3.16667563e-16, -2.51194446e00, 1.21911191e-15, -5.06049410e-16],[-2.22121281e-01, 1.17886295e-15, -1.21501454e00, -2.54198372e-02],[-2.86068006e-02, -6.44098020e-16, -2.54198372e-02, -1.10480366e00],])
hpqrs = np.array([[[[6.96056671e-01, 8.64123979e-17, 7.51963671e-02, 9.51662480e-03],[1.71207496e-16, 4.07411355e-02, -1.45820198e-17, -3.87762611e-18],[7.51963671e-02, -1.67452401e-17, 3.05202191e-02, 5.34230841e-03],[9.51662480e-03, 2.61368866e-17, 5.34230841e-03, 1.67384743e-02],],[[1.71207496e-16, 4.07411355e-02, -1.45820198e-17, -3.87762611e-18],[6.50690725e-01, 1.23045003e-17, 7.46798397e-02, 1.01652275e-02],[-2.47658344e-16, 2.43546325e-03, -4.51712874e-17, 2.48872553e-17],[2.25988600e-16, 1.23962208e-03, 6.53416581e-17, -4.61937088e-17],],[[7.51963671e-02, -1.67452401e-17, 3.05202191e-02, 5.34230841e-03],[-2.47658344e-16, 2.43546325e-03, -4.51712874e-17, 2.48872553e-17],[3.54579407e-01, -3.60619099e-17, 5.55938827e-03, -2.76946416e-03],[1.02943160e-02, 2.41224973e-17, -5.75780811e-03, -2.10651109e-02],],[[9.51662480e-03, 2.61368866e-17, 5.34230841e-03, 1.67384743e-02],[2.25988600e-16, 1.23962208e-03, 6.53416581e-17, -4.61937088e-17],[1.02943160e-02, 2.41224973e-17, -5.75780811e-03, -2.10651109e-02],[3.41425718e-01, -2.31491583e-17, 3.85389582e-03, 7.06935786e-03],],],[[[8.64123979e-17, 6.50690725e-01, -2.78377815e-16, 2.49516124e-16],[4.07411355e-02, 6.27787581e-18, 2.43546325e-03, 1.23962208e-03],[-1.67452401e-17, 7.46798397e-02, -5.60235836e-17, 6.70915608e-17],[2.61368866e-17, 1.01652275e-02, 2.78665353e-17, -5.04507066e-17],],[[4.07411355e-02, 6.27787581e-18, 2.43546325e-03, 1.23962208e-03],[1.23045003e-17, 7.62241410e-01, -3.05861696e-16, 3.55270558e-16],[2.43546325e-03, -3.26993657e-16, 1.24617768e-02, -3.33498976e-06],[1.23962208e-03, 3.18178705e-16, -3.33498976e-06, 7.25183601e-03],],[[-1.67452401e-17, 7.46798397e-02, -5.60235836e-17, 6.70915608e-17],[2.43546325e-03, -3.26993657e-16, 1.24617768e-02, -3.33498976e-06],[-3.60619099e-17, 3.59082675e-01, -5.69105903e-17, 9.60811644e-18],[2.41224973e-17, 5.08519862e-03, -9.99415338e-17, 1.14310026e-16],],[[2.61368866e-17, 1.01652275e-02, 2.78665353e-17, -5.04507066e-17],[1.23962208e-03, 3.18178705e-16, -3.33498976e-06, 7.25183601e-03],[2.41224973e-17, 5.08519862e-03, -9.99415338e-17, 1.14310026e-16],[-2.31491583e-17, 3.44431171e-01, -2.81970002e-17, -2.33497921e-17],],],[[[7.51963671e-02, -2.78377815e-16, 3.54579407e-01, 1.02943160e-02],[-1.45820198e-17, 2.43546325e-03, 5.56527542e-17, 6.35026535e-17],[3.05202191e-02, -5.60235836e-17, 5.55938827e-03, -5.75780811e-03],[5.34230841e-03, 2.78665353e-17, -2.76946416e-03, -2.10651109e-02],],[[-1.45820198e-17, 2.43546325e-03, 5.56527542e-17, 6.35026535e-17],[7.46798397e-02, -3.05861696e-16, 3.59082675e-01, 5.08519862e-03],[-4.51712874e-17, 1.24617768e-02, -5.24368327e-17, -1.09398900e-16],[6.53416581e-17, -3.33498976e-06, 3.62832929e-17, 9.42506062e-17],],[[3.05202191e-02, -5.60235836e-17, 5.55938827e-03, -5.75780811e-03],[-4.51712874e-17, 1.24617768e-02, -5.24368327e-17, -1.09398900e-16],[5.55938827e-03, -5.69105903e-17, 3.12395481e-01, 1.60872665e-02],[-5.75780811e-03, -9.99415338e-17, 1.60872665e-02, 6.97375883e-02],],[[5.34230841e-03, 2.78665353e-17, -2.76946416e-03, -2.10651109e-02],[6.53416581e-17, -3.33498976e-06, 3.62832929e-17, 9.42506062e-17],[-5.75780811e-03, -9.99415338e-17, 1.60872665e-02, 6.97375883e-02],[3.85389582e-03, -2.81970002e-17, 2.94109098e-01, -1.19378613e-02],],],[[[9.51662480e-03, 2.49516124e-16, 1.02943160e-02, 3.41425718e-01],[-3.87762611e-18, 1.23962208e-03, 6.35026535e-17, -4.14362821e-17],[5.34230841e-03, 6.70915608e-17, -5.75780811e-03, 3.85389582e-03],[1.67384743e-02, -5.04507066e-17, -2.10651109e-02, 7.06935786e-03],],[[-3.87762611e-18, 1.23962208e-03, 6.35026535e-17, -4.14362821e-17],[1.01652275e-02, 3.55270558e-16, 5.08519862e-03, 3.44431171e-01],[2.48872553e-17, -3.33498976e-06, -1.09398900e-16, -2.46748276e-18],[-4.61937088e-17, 7.25183601e-03, 9.42506062e-17, -4.91354497e-18],],[[5.34230841e-03, 6.70915608e-17, -5.75780811e-03, 3.85389582e-03],[2.48872553e-17, -3.33498976e-06, -1.09398900e-16, -2.46748276e-18],[-2.76946416e-03, 9.60811644e-18, 1.60872665e-02, 2.94109098e-01],[-2.10651109e-02, 1.14310026e-16, 6.97375883e-02, -1.19378613e-02],],[[1.67384743e-02, -5.04507066e-17, -2.10651109e-02, 7.06935786e-03],[-4.61937088e-17, 7.25183601e-03, 9.42506062e-17, -4.91354497e-18],[-2.10651109e-02, 1.14310026e-16, 6.97375883e-02, -1.19378613e-02],[7.06935786e-03, -2.33497921e-17, -1.19378613e-02, 3.06238461e-01],],],])

angles_even_real = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
angles_even_imag = np.array([3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173])
angles_odd_real = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
angles_odd_imag = np.array([3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691])



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
# Given the QKSD ground-state we want to prepare [we haven't said why we want to prepare it]
# and the polynomials [polynomial or polynomials?] which prepare the state, we can use QSP
# to initialize the state. For this we need to create the iterative quantum circuit:


def reflection(wire, ctrl_wires):
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))
    qml.Z(wire)
    qml.ctrl(qml.X(wire), ctrl_wires, [0] * len(ctrl_wires))

def rotation_about_reflection_axis(angle, wires):
    qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))
    qml.RZ(angle, wires[0])
    qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))

def qsp(lcu, angles, rot_wires, prep_wires):
    for angle in angles[::-1][:-1]:
        rotation_about_reflection_axis(angle, rot_wires)
        qml.PrepSelPrep(lcu, control=prep_wires)
    rotation_about_reflection_axis(angles[0], rot_wires)

def qsp_poly_complex(lcu, angles_real, angles_imag, ctrl_wire, rot_wires, prep_wires):
    qml.H(ctrl_wire)
    qml.ctrl(qsp, ctrl_wire, 0)(lcu, angles_real, rot_wires, prep_wires)
    qml.ctrl(qsp, ctrl_wire, 1)(lcu, angles_imag, rot_wires, prep_wires)
    qml.H(ctrl_wire)

dev = qml.device("lightning.qubit")

@qml.qnode(dev)
def krylov_qsp(lcu, rdm_ctrl_wire, ctrl_wires, rot_wires, prep_wires, measure=0, obs=None):
    if measure==2: # preprocessing for reflection measurement
        qml.X(rdm_ctrl_wire)
    
    # QSP
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



num_ancillae = int(np.log(len(coeffs)) / np.log(2)) + 1

start_wire = hamiltonian.wires[-1] + 1
rot_wires = list(range(start_wire, start_wire + num_ancillae + 1))
prep_wires = rot_wires[1:]
ctrl_wires = [prep_wires[-1] + 1, prep_wires[-1] + 2]
rdm_ctrl_wire = ctrl_wires[-1] + 1

######################################################################
# We can use these to build 1- and 2- particle reduced density matrices
# Based on [Molecular Properties from Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2501.05286)

obs = qml.H(hamiltonian.wires[-1]) @ qml.H(hamiltonian.wires[-2])

# measurement_1 = krylov_qsp(hamiltonian, rdm_ctrl_wire, ctrl_wires, rot_wires, prep_wires, measure=1, obs=obs)
# measurement_2 = krylov_qsp(hamiltonian, rdm_ctrl_wire, ctrl_wires, rot_wires, prep_wires, measure=2, obs=obs)

# print("meas 1:", measurement_1)
# print("meas 2:", measurement_2)

lambda_lcu = np.sum(np.abs(coeffs))
# coherent_result = 2*lambda_lcu*(measurement_1+measurement_2)
# print("coherent result:",coherent_result)

######################################################################
# [Explanation about the results]
#
# We can analyze the resources and flow of this program by using the qualtran call graph.
# We first convert the PennyLane circuit to a Qualtran bloq and then use the call graph to count
# the required gates:

# redefining Identity as Global Phase in hamiltonian as a temprorary work-around 
hamiltonian = qml.Hamiltonian(hamiltonian.terms()[0], [qml.GlobalPhase(0,wires=hamiltonian.wires[0]) if i.name=="Identity" else i for i in hamiltonian.terms()[1]])
bloq = qml.to_bloq(krylov_qsp, map_ops=False, lcu=hamiltonian, rdm_ctrl_wire=rdm_ctrl_wire, ctrl_wires=ctrl_wires, rot_wires=rot_wires, prep_wires=prep_wires, measure=1, obs=obs)

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

bloq = qml.to_bloq(krylov_qsp, map_ops=False, lcu=hamiltonian, rdm_ctrl_wire=rdm_ctrl_wire, ctrl_wires=ctrl_wires, rot_wires=rot_wires, prep_wires=prep_wires, measure=1, obs=obs)
graph, sigma = bloq.call_graph(generalizer=generalize_rotation_angle)
print("--- Gate counts ---")
for gate, count in sigma.items():
    print(f"{gate}: {count}")

######################################################################
# We can plot how e.g. the number of CNOT gates increases with the Krylov dimension as follows:

def count_cnots(krylov_dimension):
    angles_even_real = angles_even_imag = angles_odd_real = angles_odd_imag = np.random.random(krylov_dimension)
    bloq = qml.to_bloq(krylov_qsp, map_ops=False, lcu=hamiltonian, rdm_ctrl_wire=rdm_ctrl_wire, ctrl_wires=ctrl_wires, rot_wires=rot_wires, prep_wires=prep_wires, measure=1, obs=obs)

######################################################################
# We can also visualize the call graph to see 
#``show_call_graph``
# function to draw
######################################################################
#Placeholder
#
#
######################################################################
# Extra explanation about QKSP
#
#  .. math:: |\psi_k\rangle = U_{qsp(k)}|\psi_{hf}\rangle
#
# We then project the Jordan-Wigner Hamiltonian, :math:`\hat{H}`, into the subspace by
# [quantum krylov subspace diagonalization from https://arxiv.org/pdf/2407.14431] and obtain :math:`\tilde{H}`.
#  
# We also calculate :math:`\tilde{S}` by [insert method].
#
# We then solve the following generalized eigenvalue problem 
# 
# .. math:: \tilde{H}c^m = E_m\tilde{S}c^m
#
# 
# 
# We provide the pre-calculated values for an :math:`H_2O` molecule with Krylov dimension 15:
#
# [add example and explanation of precalculated values]
# 

######################################################################
# Conclusion
# ----------
# In this demo, we did something.
#
# About the author
# ----------------
#
