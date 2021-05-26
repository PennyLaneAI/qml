r"""
Classical Shadows
=================
.. meta::
    :property="og:description": Learn how to efficiently make predictions about an unkown 
        quantum state using the classical shadow approximation. 
    :property="og:image": https://pennylane.ai/qml/_images/qaoa_layer.png

.. related::

*Authors: Roeland Wiersema & Brian Doolittle (Xanadu Residents).
Posted: 7 May 2021. Last updated: 7 May 2021.*

Estimating properties of unknown quantum states is a key objective of quantum
information science and technology.
For example, one might want to check whether an apparatus prepares a particular target state,
or verify that an unknown system can prepare entangled states.
In principle, any unknown quantum state can be fully characterized by performing `quantum state
tomography <http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf>`_
However this procedure requires one to acquire accurate expectation values for a set of observables
which grows exponentially with the number of qubits.
A potential workaround for these scaling conderns is provided by classical shadow approximation
introduced in `Predicting many properties of a quantum system from very few measurements
<https://arxiv.org/pdf/2002.08953.pdf>`_ [Huang2020]_.

The classical shadow approximation is an efficient protocol for constructing a *classical shadow*
representation of an unknown quantum state.
The classical shadow can then be used to estimate
quantum state fidelity, Hamilton eigenvalues, two-point correlators, and many other properties.

.. figure:: ../demonstrations/classical_shadows/classical_shadow_overview.png
    :align: center
    :width: 80%

    (Image from Huang et al. [Huang2020]_.)

In this demo, we will show how to construct classical shadows and use them to approximate
properties of quantum states.
To do this, we will take a test oriented approach where we develop source code, test it, and
demo it in a jupyter notebook.
In this demo, we will work in three separate files:

* ``./classical_shadows.py`` - source code for the classical shadow approximation.
* ``./test_classical_shadows.py`` - test code for ``./classical_shadows.py``.
* ``./notebook_classical_shadows.ipynb`` - jupyter notebook for demoing classical shadows.

For clarity, this demo will specify the file in each respective code block.



First create the ``./classical_shadows.py`` file and import the following libraries.
"""

# ./classical_shadows.py
import pennylane as qml
import pennylane.numpy as np
from typing import List

##############################################################################
# Constructing a Classical Shadow
# ###########################################
#
# A classical shadow consists of an integer number `N` *snapshots* which are constructed
# via the following proce
#
# 1. A quantum state :math:`\rho` is prepared.
# 2. A randomly selected unitary :math:`U` is applied
# 3. A computational basis measurement is performed.
# 4. The process is repeated :math:`N` times.
#
# The classical shadow is then constructed as a list of measurement outcomes and chosen unitaries.

# ./classical_shadows.py

def calculate_classical_shadow(circuit_template, params, shadow_size: int, num_qubits: int) -> np.ndarray:
    """
    Given a circuit, creates a collection of snapshots U^\\dag|b><b| U with the stabilizer description.
    
    Args:
        circuit_template: A Pennylane QNode.
        params: Circuit parameters.
        shadow_size: The number of snapshots in the shadow.
        num_qubits: The number of qubits in the circuit.

    Returns:
        Numpy array containing the outcomes (0, 1) in the first `num_qubits` columns and the sampled Pauli's
        (0,1,2=x,y,z) in the final `num_qubits` columns.
    """

    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    # each shadow is one shot, so we set this parameter in the qml.device
    # sample random pauli unitaries uniformly, where 1,2,3 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))
    for ns in range(shadow_size):
        # for each shadow, add a random Clifford observable at each location
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, wires=num_qubits, observable=obs)
    # combine the computational basis outcomes and the sampled unitaries
    return np.concatenate([outcomes, unitary_ids], axis=1)

##############################################################################
# To test the ``calculate_classical_shadow`` function first create a file called
# ``test_classical_shadows.py`` and import the following libraries.

# ./test_classical_shadows.py

import pytest
import pennylane as qml
import pennylane.numpy as np
import time

##############################################################################
#
# Furthermore, we'll need to import the functions from the source code::
#
#     # ./test_classical_shadows.py
#
#     from classical_shadows import calculate_classical_shadow, estimate_shadow_obervable, shadow_state_reconstruction, operator_2_norm, shadow_bound
#
# Now we're ready to start writing tests.

# ./test_classical_shadows.py

@pytest.fixture
def circuit_1_observable(request):
    """Circuit with single layer requiring nqubits parameters"""
    num_qubits = request.param
    dev = qml.device('default.qubit', wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, wires, **kwargs):
        observables = kwargs.pop('observable')
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return [qml.expval(o) for o in observables]

    param_shape = (None,)
    return circuit, param_shape, num_qubits


@pytest.fixture
def circuit_1_state(request):
    """Circuit with single layer requiring nqubits parameters"""
    num_qubits = request.param
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(device=dev)
    def circuit(params, wires, **kwargs):
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return qml.state()

    param_shape = (None,)
    return circuit, param_shape, num_qubits


@pytest.fixture
def circuit_2_observable(request):
    """Circuit with multiple layers requiring nqubits*3 parameters"""
    num_qubits = request.param
    dev = qml.device('default.qubit', wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, wires, **kwargs):
        observables = kwargs.pop('observable')
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w, 0], wires=w)
        for layer in range(2):
            for w in dev.wires:
                qml.RX(params[w, layer + 1], wires=w)
        return [qml.expval(o) for o in observables]

    param_shape = (None, 3)
    return circuit, param_shape, num_qubits


# TODO: Do both circuit_1 and circuit_2 in the same test.
# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_1_observable", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_1(circuit_1_observable, shadow_size=10):
    """Test calculating the shadow for a simple circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    outcomes = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    print(shadow)


    assert all(o in [1.0, -1.0] for o in np.unique(outcomes[:, :num_qubits]))


# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_2_observable", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_2(circuit_2_observable, shadow_size=10):
    """Test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)


    assert all(o in [1.0, -1.0] for o in np.unique(shadow[:, :num_qubits]))

@pytest.mark.parametrize("circuit_1_observable, shadow_size", [[2, 10], [2, 100], [2, 1000], [2, 10000]],
                         indirect=['circuit_1_observable'])
def test_calculate_classical_shadow_performance(circuit_1_observable, shadow_size):
    """Performance test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    start = time.time()
    calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)
    delta_time = time.time() - start
    print(f'Elapsed time for {shadow_size} shadows = {delta_time}')

##############################################################################
# To run the tests, simply use the command ``$ pytest ./test_classical_shadows.py``.
# The output of the tests should look something like this:

##############################################################################
# Estimating Pauli Observables with Classical Shadows
# ###################################################
#
# Any pauli observable can estimated by using the ``estimate_shadow_observable`` method.

# ./classical_shadows.py
def estimate_shadow_obervable(shadows, observable) -> float:
    """
    Calculate the estimator E[O] = sum_i Tr{rho_i O} where rho_i is a snapshot in the shadow.
    
    Args:
        shadows: Numpy array containing the outcomes (0, 1) in the first `num_qubits` columns and the sampled Pauli's
        (0,1,2=x,y,z) in the final `num_qubits` columns.
        observable: Single PennyLane observable consisitng of single Pauli operators e.g. qml.PauliX(0) @ qml.PauliY(1)
    
    Returns:
        Scalar corresponding to the estimate of the observable.
    """
    map_name_to_int = {'PauliX': 0, 'PauliY': 1, 'PauliZ': 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        observable_as_list = [(map_name_to_int[observable.name], observable.wires[0])]
    else:
        observable_as_list = [(map_name_to_int[o.name], o.wires[0]) for o in observable.obs]

    num_qubits = shadows.shape[1] // 2
    sum_product, cnt_match = 0, 0
    # loop over the shadows:
    for single_measurement in shadows:
        not_match = 0
        product = 1
        # loop over all the paulis that we care about
        for pauli_XYZ, position in observable_as_list:
            # if the pauli in our shadow does not match, we break and go to the next shadow
            if pauli_XYZ != single_measurement[position + num_qubits]:
                not_match = 1
                break
            product *= single_measurement[position]
        # don't record the shadow
        if not_match == 1: continue

        sum_product += product
        cnt_match += 1
    if cnt_match == 0:
        return 0
    else:
        return sum_product / cnt_match


##############################################################################
# Not sure where these utility methods actually belong in the demo.

# ./classical_shadows.py
def operator_2_norm(R):
    """
    Calculate the operator two norm.
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))


def shadow_bound(M: int, error: float, max_k: int, observables: List[np.ndarray]):
    """
    Calculate the shadow bound for the pauli measurement scheme.
    """
    shadow_norm = lambda op: np.linalg.norm(op, ord=np.inf) ** 2
    return int(np.ceil(np.log(M) * 4 ** max_k * max(shadow_norm(o) for o in observables) / error ** 2))

##############################################################################
# Testing ``estimate_shadow_observable``

# ./test_classical_shadows.py

# TODO: create a fixture for the shadow so we only have run it once
@pytest.mark.parametrize("circuit_1_observable, shadow_size", [[2, 10], [2, 100], [2, 1000], [2, 10000]],
                         indirect=['circuit_1_observable'])
def test_estimate_shadow_observable_single(circuit_1_observable, shadow_size):
    """Test calculating an observable with the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    observable = [qml.PauliX(0) @ qml.PauliX(1)]
    expval_shadow = estimate_shadow_obervable(shadow, observable[0])
    assert -1.0 <= expval_shadow <= 1.0
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = circuit_template(params, wires=dev_exact.wires, observable=observable)
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")

@pytest.mark.parametrize("circuit_2_observable", [8], indirect=['circuit_2_observable'])
def test_estimate_shadow_observable_shadow_bound(circuit_2_observable):
    """Test calculating multiple observables with the shadowsize shadow_size given by the bound in the paper"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    list_of_observables = [qml.PauliX(0) @ qml.PauliZ(2) @ qml.PauliY(4),
                           qml.PauliY(1) @ qml.PauliZ(2),
                           qml.PauliX(0),
                           qml.PauliY(3) @ qml.PauliZ(4) @ qml.PauliY(num_qubits - 1)]
    # Calculate how many shadows we need to get an error of 1e-1 for this set of observables
    shadow_size = shadow_bound(M=len(list_of_observables), max_k=3,
                               error=1e-1, observables=[o.matrix for o in list_of_observables])

    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    expval_shadow = sum(estimate_shadow_obervable(shadow, o) for o in list_of_observables)
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = sum(circuit_template(params, wires=dev_exact.wires, observable=[o, ]) for o in list_of_observables)
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")
    # from the theoretical bound we know that the variance of this estimator must be within 1e-1
    assert np.isclose(expval_shadow, expval_exact, atol=1e-1)


##############################################################################
# To demonstrate estimation with classical shadows, we'll use  
# the simple example of estimating the observable :math:`X_0' with
# classical shadows.

# ./notebook_classical_shadows.ipynb
nqubits = 1
theta = np.random.randn(nqubits, )
device_exact = qml.device('default.qubit', wires=nqubits)
number_of_shadows = 1000

paulis = {0: qml.Identity(0).matrix,
          1: qml.PauliX(0).matrix,
          2: qml.PauliY(0).matrix,
          3: qml.PauliZ(0).matrix}


def circuit(params, wires, **kwargs):
    for i in range(nqubits):
        qml.Hadamard(wires=i)
        qml.RY(params[i], wires=i)


# Estimate Tr{X_0 rho}
exact_observable = qml.map(circuit, observables=[qml.PauliX(0)],
                           device=device_exact, measure='expval')(theta)
print(f"Exact value: {exact_observable[0]}")
shadows = calculate_classical_shadow(circuit, theta, number_of_shadows, nqubits)
# The observable X_0 is passed as [(1,0), ] something like X_0 Y_1 would be  [(1,0),(2,1)]
# TODO: Generalize the mapping from Pauli strings to tuples
shadow_observable = estimate_shadow_obervable(shadows, qml.PauliX(0))
print(f"Shadow value: {shadow_observable}")
shadows

##############################################################################
# Next, we consider the multi-qubit observable :math:`X_0 X_1`.


# ./notebook_classical_shadows.ipynb
nqubits = 2
theta = np.random.randn(nqubits, )
device_exact = qml.device('default.qubit', wires=nqubits)
number_of_shadows = 3000

exact_observable = qml.map(circuit, observables=[qml.PauliX(0) @ qml.PauliX(1)],
                           device=device_exact, measure='expval')(
    theta)
print(f"Exact value: {exact_observable[0]}")
shadows = calculate_classical_shadow(circuit, theta, number_of_shadows, nqubits)
shadow_observable = estimate_shadow_obervable(shadows, qml.PauliX(0) @ qml.PauliX(1))
print(f"Shadow value: {shadow_observable}")

exact_observable

##############################################################################
# Comparison of standard observable estimators and classical shadows.

##############################################################################
# State Reconstruction using Classical Shadow
# ###########################################

# ./classical_shadows.py
def snapshot_state(b_list, obs_list):
    """
    Reconstruct a state approximation from a single snapshot in a shadow.
    **Details:**
    Implements Eq. (S44) from https://arxiv.org/pdf/2002.08953.pdf
    Args:
        b_list (array): classical outcomes for a single sample
        obs_list (array): ids for the pauli observable used for each measurement
    """

    num_qubits = len(b_list)

    paulis = [
        qml.Hadamard(0).matrix,

        qml.Hadamard(0).matrix @ np.array([[1, 0],
                                           [0, -1j]], dtype=np.complex),
        qml.Identity(0).matrix
    ]

    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = paulis[int(obs_list[i])]

        local_rho = 3 * (U.conj().T @ state @ U) - np.eye(2, 2)

        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


##############################################################################
# Example: |+> state is approximated as maximally mixed state

# ./classical_shadows.py
def shadow_state_reconstruction(shadow):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.
    """

    num_shadows = shadow.shape[0]
    num_qubits = shadow.shape[1] // 2

    b_lists = shadow[:, 0:num_qubits]
    obs_lists = shadow[:, num_qubits:2 * num_qubits]
    # state approximated from snapshot average
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=np.complex)
    for i in range(num_shadows):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_shadows

##############################################################################
# Test shadow state reconstruction

# ./test_classical_shadows.py
@pytest.mark.parametrize("circuit_1_observable, circuit_1_state, shadow_size",
                         [[2, 2, 1000], [2, 2, 5000]], indirect=['circuit_1_observable',
                                                                                 'circuit_1_state'])
def test_shadow_state_reconstruction(circuit_1_observable, circuit_1_state, shadow_size):
    """Test reconstructing the state from the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    circuit_template_state, _, _ = circuit_1_state
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    state_shadow = shadow_state_reconstruction(shadow)
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    circuit_template.device = dev_exact
    state_exact = circuit_template_state(params, wires=dev_exact.wires, observable=[qml.state()])
    state_exact = np.outer(state_exact, state_exact.conj())

    print(operator_2_norm(state_shadow - state_exact))


##############################################################################
# Example: |+> state is approximated as maximally mixed state

# ./notebook_classical_shadows.ipynb
nqubits = 1
theta = [np.pi/4]
number_of_shadows = 500

def one_qubit_RY(params, wires, **kwargs):
    qml.RY(params[0], wires=wires)

shadow = calculate_classical_shadow(one_qubit_RY, theta, number_of_shadows, nqubits)

state_reconstruction = shadow_state_reconstruction(shadow)

print("state reconstruction")

print(state_reconstruction)



#TODO: what is a good application?

##############################################################################
# .. [Huang2020] Huang, Hsin-Yuan, Richard Kueng, and John Preskill.
#             "Predicting many properties of a quantum system from very few measurements."
#             Nature Physics 16.10 (2020): 1050-1057.

