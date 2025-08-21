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
H, qubits = qml.qchem.molecular_hamiltonian(molecule)
print(H)

######################################################################
# Many subspaces are possible, but in this case we choose the span of the first 15 Chebyshev
# polynomials of the Hamiltonian applied to the Hartree-Fock ground state: 
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
# Using a Krylov subspace of dimension :math:`D=15`, we pre-computed the required values [what values?] for the
# :math:`H_2O` molecule. This can be done using e.g. the [lanczos method](https://quantum-journal.org/papers/q-2023-05-23-1018/pdf/)
# [is this correct?] Or using [Quantum Krylov Subspace diagonalization](https://arxiv.org/pdf/2407.14431).
# 
# 
# We provide the pre-calculated values for an :math:`H_2O` molecule with Krylov dimension 15:


import numpy as np
import pennylane as qml
from scipy.stats import ortho_group


class QubitizedWalkOperator:
    @classmethod
    def preprocess_lcu(cls, lcu_coeffs, lcu_paulis):
        assert np.isclose(np.sum(lcu_coeffs**2), 1.0)

        for i, v in enumerate(lcu_coeffs):
            if v < 0:
                lcu_coeffs[i] = np.abs(v)
                lcu_paulis[i] = -lcu_paulis[i]

        return lcu_coeffs, lcu_paulis

    def __init__(
        self, lcu_coeffs, lcu_paulis, lambda_lcu, prep_wires, reflection_wires
    ) -> None:
        self._lcu_coeffs, self._lcu_paulis = QubitizedWalkOperator.preprocess_lcu(
            lcu_coeffs, lcu_paulis
        )
        self._lambda_lcu = lambda_lcu
        self.prep_wires = prep_wires
        self.reflection_wires = reflection_wires

    def block_encode_circuit(self):
        qml.StatePrep(self._lcu_coeffs, wires=self.prep_wires)
        qml.Select(self._lcu_paulis, control=self.prep_wires)
        qml.adjoint(qml.StatePrep(self._lcu_coeffs, wires=self.prep_wires))

    def reflection_circuit(self):
        # qml.PauliX(self.reflection_wires[0])
        qml.ctrl(
            qml.PauliX(self.reflection_wires[0]),
            self.reflection_wires[1:],
            (0,) * len(self.reflection_wires[1:]),
        )
        qml.PauliZ(self.reflection_wires[0])
        qml.ctrl(
            qml.PauliX(self.reflection_wires[0]),
            self.reflection_wires[1:],
            (0,) * len(self.reflection_wires[1:]),
        )
        # qml.PauliX(self.reflection_wires[0])

    def qubitized_walk_operator(self):
        qml.PauliX(self.reflection_wires[0])
        self.block_encode_circuit()
        self.reflection_circuit()
        qml.PauliX(self.reflection_wires[0])


def lcu_decomp(A, wire_order):
    lcu = qml.pauli_decompose(A, wire_order=wire_order)
    coeffs, pls = lcu.terms()

    for i, v in enumerate(coeffs):
        if v < 0:
            coeffs[i] = np.abs(v)
            pls[i] = -pls[i]

    n = int(np.log(len(coeffs)) / np.log(2)) + 1
    print(n)
    coeffs = np.array(list(coeffs) + [0] * (2**n - len(coeffs)))
    print(len(coeffs))
    print(pls)
    lambda_lcu = np.sum(coeffs)
    normalized_coeffs = np.sqrt(coeffs / lambda_lcu)
    return normalized_coeffs, pls, lambda_lcu


def transform_into_proper_lcu(coeffs, pls):
    """set the coeffs to their absolute value and include the sign into the Pauli string

    Args:
        coeffs (): LCU coefficients
        pls (): Pauli strings

    Returns:
    """

    for i, v in enumerate(coeffs):
        if v < 0:
            coeffs[i] = np.abs(v)
            pls[i] = -pls[i]

    n = int(np.log(len(coeffs)) / np.log(2)) + 1
    print(n)
    coeffs = np.array(list(coeffs) + [0] * (2**n - len(coeffs)))
    print(len(coeffs))
    print(pls)
    lambda_lcu = np.sum(coeffs)
    normalized_coeffs = np.sqrt(coeffs / lambda_lcu)
    return normalized_coeffs, pls, lambda_lcu


def shift_and_transform_of_qubit_op_to_PL_lcu(qubit_op, ancilla_shift):
    """shift the qubit operator to act on the proper target qubits and return the according PL LCU

    Args:
        qubit_op ():
        ancilla_shift ():

    Returns:
    """
    shifted_qubit_operator = {}

    for key in qubit_op.keys():
        if key == ():
            shifted_key = key
        else:
            shifted_key = ()
            for k in key:
                shifted_key += ((k[0] + ancilla_shift, k[1]),)

        shifted_qubit_operator[shifted_key] = qubit_op[key]
    PL_ops_list = []

    for key in shifted_qubit_operator.keys():
        PL_op_list = []
        if key == ():
            PL_op_list.append(qml.Identity())
        else:
            for k in key:
                if k[1] == "X":
                    PL_op_list.append(qml.PauliX(k[0]))

                if k[1] == "Y":
                    PL_op_list.append(qml.PauliY(k[0]))

                if k[1] == "Z":
                    PL_op_list.append(qml.PauliZ(k[0]))

        PL_ops_list.append(qml.prod(*PL_op_list))

    return transform_into_proper_lcu(list(shifted_qubit_operator.values()), PL_ops_list)


class ChebychevQubitizedOperator(QubitizedWalkOperator):
    def __init__(
        self, lcu_coeffs, lcu_paulis, lambda_lcu, prep_wires, reflection_wires, order
    ) -> None:
        super().__init__(
            lcu_coeffs, lcu_paulis, lambda_lcu, prep_wires, reflection_wires
        )
        self.order = order

    def chebychev_operator_circuit(self):
        for _ in range(self.order):
            self.qubitized_walk_operator()


def generate_decoy_electron_integrals(n):
    hpq = np.random.normal(size=(n, n))
    hpq = 0.5 * (hpq + hpq.T)
    ut = np.array([ortho_group.rvs(n) for _ in range(n * (n - 1) // 2)])
    zt = [np.random.normal(size=(n,)) for _ in range(len(ut))]
    zt = np.array([np.abs(np.random.normal(size=1)) * np.outer(x, x) for x in zt])
    hpqrs = np.einsum("tpk,tqk,tkl,trl,tsl->pqrs", ut, ut, zt, ut, ut)
    hpqrs = np.transpose(hpqrs, (0, 2, 1, 3))
    return hpq, hpqrs


class QSPKrylovState(ChebychevQubitizedOperator):
    def __init__(
        self,
        lcu_coeffs,
        lcu_paulis,
        lambda_lcu,
        prep_wires,
        reflection_wires,
        rot_wires,
        control_wires,
        order,
        angles_odd,
        angles_conj_odd,
        angles_even,
        angles_conj_even,
    ) -> None:
        super().__init__(
            lcu_coeffs, lcu_paulis, lambda_lcu, prep_wires, reflection_wires, order
        )

        self.angles_odd = angles_odd
        self.angles_conj_odd = angles_conj_odd
        self.angles_even = angles_even
        self.angles_conj_even = angles_conj_even
        self.rot_wires = rot_wires
        self.control_wires = control_wires

    @staticmethod
    def rotation_around_refl_ax(phi, wires):
        qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))
        qml.RZ(phi, wires[0])
        qml.ctrl(qml.PauliX(wires[0]), wires[1:], (0,) * len(wires[1:]))

    def iterate(self, phi):
        __class__.rotation_around_refl_ax(phi, self.rot_wires)
        self.block_encode_circuit()

    def qsp_building_template(self, phis):
        for a in phis[::-1][:-1]:
            self.iterate(phi=a)
        __class__.rotation_around_refl_ax(phis[0], self.rot_wires)

    def qsp_poly_with_parity(self, angles, angles_conj):
        qml.Hadamard(self.control_wires[1])
        qml.ctrl(self.qsp_building_template, self.control_wires[1], 0)(angles)
        qml.ctrl(self.qsp_building_template, self.control_wires[1], 1)(angles_conj)
        qml.Hadamard(self.control_wires[1])

    def qsp_generic_polynomial(self):
        qml.Hadamard(self.control_wires[0])
        qml.ctrl(self.qsp_poly_with_parity, self.control_wires[0], 0)(
            self.angles_even, self.angles_conj_even
        )
        qml.ctrl(self.qsp_poly_with_parity, self.control_wires[0], 1)(
            self.angles_odd, self.angles_conj_odd
        )
        qml.Hadamard(self.control_wires[0])

hpq = np.array([[-2.50986452e00, 1.05657350e-16, -2.22121281e-01, -2.86068006e-02],[3.16667563e-16, -2.51194446e00, 1.21911191e-15, -5.06049410e-16],[-2.22121281e-01, 1.17886295e-15, -1.21501454e00, -2.54198372e-02],[-2.86068006e-02, -6.44098020e-16, -2.54198372e-02, -1.10480366e00],])
hpqrs = np.array([[[[6.96056671e-01, 8.64123979e-17, 7.51963671e-02, 9.51662480e-03],[1.71207496e-16, 4.07411355e-02, -1.45820198e-17, -3.87762611e-18],[7.51963671e-02, -1.67452401e-17, 3.05202191e-02, 5.34230841e-03],[9.51662480e-03, 2.61368866e-17, 5.34230841e-03, 1.67384743e-02],],[[1.71207496e-16, 4.07411355e-02, -1.45820198e-17, -3.87762611e-18],[6.50690725e-01, 1.23045003e-17, 7.46798397e-02, 1.01652275e-02],[-2.47658344e-16, 2.43546325e-03, -4.51712874e-17, 2.48872553e-17],[2.25988600e-16, 1.23962208e-03, 6.53416581e-17, -4.61937088e-17],],[[7.51963671e-02, -1.67452401e-17, 3.05202191e-02, 5.34230841e-03],[-2.47658344e-16, 2.43546325e-03, -4.51712874e-17, 2.48872553e-17],[3.54579407e-01, -3.60619099e-17, 5.55938827e-03, -2.76946416e-03],[1.02943160e-02, 2.41224973e-17, -5.75780811e-03, -2.10651109e-02],],[[9.51662480e-03, 2.61368866e-17, 5.34230841e-03, 1.67384743e-02],[2.25988600e-16, 1.23962208e-03, 6.53416581e-17, -4.61937088e-17],[1.02943160e-02, 2.41224973e-17, -5.75780811e-03, -2.10651109e-02],[3.41425718e-01, -2.31491583e-17, 3.85389582e-03, 7.06935786e-03],],],[[[8.64123979e-17, 6.50690725e-01, -2.78377815e-16, 2.49516124e-16],[4.07411355e-02, 6.27787581e-18, 2.43546325e-03, 1.23962208e-03],[-1.67452401e-17, 7.46798397e-02, -5.60235836e-17, 6.70915608e-17],[2.61368866e-17, 1.01652275e-02, 2.78665353e-17, -5.04507066e-17],],[[4.07411355e-02, 6.27787581e-18, 2.43546325e-03, 1.23962208e-03],[1.23045003e-17, 7.62241410e-01, -3.05861696e-16, 3.55270558e-16],[2.43546325e-03, -3.26993657e-16, 1.24617768e-02, -3.33498976e-06],[1.23962208e-03, 3.18178705e-16, -3.33498976e-06, 7.25183601e-03],],[[-1.67452401e-17, 7.46798397e-02, -5.60235836e-17, 6.70915608e-17],[2.43546325e-03, -3.26993657e-16, 1.24617768e-02, -3.33498976e-06],[-3.60619099e-17, 3.59082675e-01, -5.69105903e-17, 9.60811644e-18],[2.41224973e-17, 5.08519862e-03, -9.99415338e-17, 1.14310026e-16],],[[2.61368866e-17, 1.01652275e-02, 2.78665353e-17, -5.04507066e-17],[1.23962208e-03, 3.18178705e-16, -3.33498976e-06, 7.25183601e-03],[2.41224973e-17, 5.08519862e-03, -9.99415338e-17, 1.14310026e-16],[-2.31491583e-17, 3.44431171e-01, -2.81970002e-17, -2.33497921e-17],],],[[[7.51963671e-02, -2.78377815e-16, 3.54579407e-01, 1.02943160e-02],[-1.45820198e-17, 2.43546325e-03, 5.56527542e-17, 6.35026535e-17],[3.05202191e-02, -5.60235836e-17, 5.55938827e-03, -5.75780811e-03],[5.34230841e-03, 2.78665353e-17, -2.76946416e-03, -2.10651109e-02],],[[-1.45820198e-17, 2.43546325e-03, 5.56527542e-17, 6.35026535e-17],[7.46798397e-02, -3.05861696e-16, 3.59082675e-01, 5.08519862e-03],[-4.51712874e-17, 1.24617768e-02, -5.24368327e-17, -1.09398900e-16],[6.53416581e-17, -3.33498976e-06, 3.62832929e-17, 9.42506062e-17],],[[3.05202191e-02, -5.60235836e-17, 5.55938827e-03, -5.75780811e-03],[-4.51712874e-17, 1.24617768e-02, -5.24368327e-17, -1.09398900e-16],[5.55938827e-03, -5.69105903e-17, 3.12395481e-01, 1.60872665e-02],[-5.75780811e-03, -9.99415338e-17, 1.60872665e-02, 6.97375883e-02],],[[5.34230841e-03, 2.78665353e-17, -2.76946416e-03, -2.10651109e-02],[6.53416581e-17, -3.33498976e-06, 3.62832929e-17, 9.42506062e-17],[-5.75780811e-03, -9.99415338e-17, 1.60872665e-02, 6.97375883e-02],[3.85389582e-03, -2.81970002e-17, 2.94109098e-01, -1.19378613e-02],],],[[[9.51662480e-03, 2.49516124e-16, 1.02943160e-02, 3.41425718e-01],[-3.87762611e-18, 1.23962208e-03, 6.35026535e-17, -4.14362821e-17],[5.34230841e-03, 6.70915608e-17, -5.75780811e-03, 3.85389582e-03],[1.67384743e-02, -5.04507066e-17, -2.10651109e-02, 7.06935786e-03],],[[-3.87762611e-18, 1.23962208e-03, 6.35026535e-17, -4.14362821e-17],[1.01652275e-02, 3.55270558e-16, 5.08519862e-03, 3.44431171e-01],[2.48872553e-17, -3.33498976e-06, -1.09398900e-16, -2.46748276e-18],[-4.61937088e-17, 7.25183601e-03, 9.42506062e-17, -4.91354497e-18],],[[5.34230841e-03, 6.70915608e-17, -5.75780811e-03, 3.85389582e-03],[2.48872553e-17, -3.33498976e-06, -1.09398900e-16, -2.46748276e-18],[-2.76946416e-03, 9.60811644e-18, 1.60872665e-02, 2.94109098e-01],[-2.10651109e-02, 1.14310026e-16, 6.97375883e-02, -1.19378613e-02],],[[1.67384743e-02, -5.04507066e-17, -2.10651109e-02, 7.06935786e-03],[-4.61937088e-17, 7.25183601e-03, 9.42506062e-17, -4.91354497e-18],[-2.10651109e-02, 1.14310026e-16, 6.97375883e-02, -1.19378613e-02],[7.06935786e-03, -2.33497921e-17, -1.19378613e-02, 3.06238461e-01],],],])
system_size = 4
def build_circuit(
    poly_coeffs, angles_even, angles_conj_even, angles_odd, angles_conj_odd, poly_degree
):
    interaction_op = InteractionOperator(
        constant=0.0, one_body_tensor=hpq, two_body_tensor=hpqrs
    )

    fermion_hamiltonian = get_fermion_operator(interaction_op)
    hamiltonian = observable(fermion_hamiltonian)
    lcu_coeffs, _ = hamiltonian.terms()

    ancilla_shift = int(np.log(len(lcu_coeffs)) / np.log(2)) + 1
    ancilla_shift += 1  # add one for the reflection
    ancilla_shift += 1  # ancilla for P,P^{*} control
    ancilla_shift += 1  # ancilla for odd/even partiy control

    # shift the Hamiltonina operator by the ancillae value
    shifted_qubit_operator = {}
    qubit_op = jordan_wigner(fermion_hamiltonian).terms
    s = sum(map(abs, qubit_op.values()))
    for key in qubit_op.keys():
        if key == ():
            shifted_key = key
        else:
            shifted_key = ()
            for k in key:
                shifted_key += ((k[0] + ancilla_shift, k[1]),)

        shifted_qubit_operator[shifted_key] = qubit_op[key]

    coeffs, pls, lambda_value = shift_and_transform_of_qubit_op_to_PL_lcu(
        qubit_op, ancilla_shift
    )
    lambda_value = float(lambda_value)

    assert np.isclose(s, lambda_value, atol=0.0, rtol=1e-8)

    generic_poly = QSPKrylovState(
        lcu_coeffs=coeffs,
        lcu_paulis=pls,
        lambda_lcu=lambda_value,
        prep_wires=range(3, ancilla_shift),
        reflection_wires=range(2, ancilla_shift),
        rot_wires=range(2, ancilla_shift),
        control_wires=range(2),
        order=poly_degree,
        angles_conj_even=angles_conj_even,
        angles_conj_odd=angles_conj_odd,
        angles_even=angles_even,
        angles_odd=angles_odd,
    )
    # Final circuit
    qsp_circuit_of_poly_with_parity = generic_poly.qsp_generic_polynomial

    if system_size < 4:
        # For sanity checks
        hamiltonian = qml.matrix(hamiltonian)
        e, u = np.linalg.eigh(hamiltonian)
        e /= lambda_value
        P_even = (
            lambda ord: u
            @ np.diag(
                [
                    sum(
                        [
                            poly_coeffs[i] * np.cos(i * np.arccos(x))
                            for i in range(0, ord + 1, 2)
                        ]
                    )
                    for x in e
                ]
            )
            @ u.T
        )
        P_odd = (
            lambda ord: u
            @ np.diag(
                [
                    sum(
                        [
                            poly_coeffs[i] * np.cos(i * np.arccos(x))
                            for i in range(1, ord + 1, 2)
                        ]
                    )
                    for x in e
                ]
            )
            @ u.T
        )

        ref = lambda_value * (P_even(poly_degree) + P_odd(poly_degree))

        be_mtx = 2 * lambda_value * qml.matrix(qsp_circuit_of_poly_with_parity)()
        print(np.max(np.abs(be_mtx[: 2**system_size, : 2**system_size] - ref)))
        assert np.allclose(np.real(be_mtx[: 2**system_size, : 2**system_size]), ref)

    return qsp_circuit_of_poly_with_parity

poly_coeffs = np.array([0.04839504, -0.11122954, -0.11047445, 0.19321127, 0.10155112, -0.07727596, 0.12920728, 0.03954071, -0.25404897, -0.03059887, -0.00703527, -0.28554963, 0.14565572, -0.1226918, 0.02780524])
angles_even = np.array([3.11277458, 2.99152757, 3.15307452, 3.40611024, 3.00166196, 3.03597059, 3.25931224, 3.04073693, 3.25931224, 3.03597059, 3.00166196, 3.40611024, 3.15307452, 2.99152757, -40.86952257])
angles_conj_even = np.array([[3.17041073, 3.29165774, 3.13011078, 2.87707507, 3.28152334, 3.24721472, 3.02387307, 3.24244837, 3.02387307, 3.24721472, 3.28152334, 2.87707507, 3.13011078, 3.29165774, -47.09507173]])
angles_odd = np.array([3.26938242, 3.43658284, 3.17041296, 3.10158929, 3.22189574, 2.93731798, 3.25959312, 3.25959312, 2.93731798, 3.22189574, 3.10158929, 3.17041296, 3.43658284, -37.57132208])
angles_conj_odd = [3.01380289, 2.84660247, 3.11277234, 3.18159601, 3.06128956, 3.34586733, 3.02359219, 3.02359219, 3.34586733, 3.06128956, 3.18159601, 3.11277234, 2.84660247, -44.11008691]
poly_degree = len(poly_coeffs) - 1

cir = build_circuit(
    poly_coeffs,
    angles_even,
    angles_conj_even,
    angles_odd,
    angles_conj_odd,
    poly_degree,
)

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
