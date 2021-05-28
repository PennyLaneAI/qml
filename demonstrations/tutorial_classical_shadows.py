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
A potential workaround for these scaling concerns is provided by the classical shadow approximation
introduced in `Predicting many properties of a quantum system from very few measurements
<https://arxiv.org/pdf/2002.08953.pdf>`_ [Huang2020]_.

The classical shadow approximation is an efficient protocol for constructing a *classical shadow*
representation of an unknown quantum state.
The classical shadow can be used to estimate properties such as
quantum state fidelity, Hamiltonian observables, and two-point correlators.

.. figure:: ../demonstrations/classical_shadows/classical_shadow_overview.png
    :align: center
    :width: 80%

    (Image from Huang et al. [Huang2020]_.)

In this demo, we will use PennyLane to construct classical shadows and use them to reconstruct
quantum states and estimate observables.
We will use a test oriented approach where we develop a ``classical_shadows.py`` library complete
with testts and Jupyter noebook examples.
In this demo, we will work in three separate files:

* ``./classical_shadows.py`` - source code for the classical shadow approximation.
* ``./test_classical_shadows.py`` - test code for ``./classical_shadows.py``.
* ``./notebook_classical_shadows.ipynb`` - jupyter notebook for demoing classical shadows.

For clarity, this demo will specify the file at the top of each code block.
"""

#####################################################################
# Constructing a Classical Shadow
# ###############################
#
# Classical shadow estimation relies on the fact that for a particular choice of circuit measurements,
# we can store an efficient representation of the state that can be used to
# predict linear functions of observables. Depending on what type of measurements we choose,
# we have an informational theoretical bound that controls the precision of our estimator.
#
# Let us consider a :math:`n`-qubit quantum state :math:`\rho` and apply a random unitary
# :math:`U` to the state:
#
# .. math::
#
#     \rho \to U \rho U^\dagger.
#
# If we measure in the computational basis, we obtain a bitstring of outcomes :math:`|b\rangle = |0011\ldots10\rangle`.
# If the unitaries :math:`U` are chosen at random from a particular ensemble, then we can store the reverse operation
# :math:`U^\dagger |b\rangle\langle b| U` efficiently in classical memory. Moreover, we can view
# the average over these snapshots of the state as a measurement channel:
#
# .. math::
#
#      \mathbb{E}\left[U^\dagger |b\rangle\langle b| U\right] = \mathcal{M}(\rho).
#
# If the ensemble of unitaries defines a tomographically complete set of measurements,
# we can invert the channel and reconstruct the state:
#
# .. math::
#
#      \rho = \mathbb{E}\left[\mathcal{M}^{-1}\left(U^\dagger |b\rangle\langle b| U \right)\right].
#
# Note that this inverted channel is not physical, i.e., it is not completely postive and trace preserving (CPTP).
# But this is of no concern to us, since all we care about is efficiently applying this inverse channel
# as a post-processing step to reconstruct the state.
#
# If we apply the procedure outlined above :math:`N` times, then the set of inverted snapshots
# is what we call the *classical shadow*
#
# .. math::
#
#      S(\rho,N) = \left\{\hat{\rho}_1= \mathcal{M}^{-1}\left(U_1^\dagger |b_1\rangle\langle b_1| U_1 \right)
#      ,\ldots, \hat{\rho}_N= \mathcal{M}^{-1}\left(U_N^\dagger |b_N\rangle\langle b_N| U_N \right)
#      \right\}.
#
# By definition, we can now estimate **any** observable as
#
# .. math::
#
#      \langle O \rangle = \sum_i \text{Tr}{\hat{\rho}_i O}.
#
# In fact, the authors prove that with a shadow of size :math:`N`, we can predict :math:`M` arbitary linear functions
# :math:`\text{Tr}{O_1\rho},\ldots,\text{Tr}{O_M \rho}` of to additive error :math:`\epsilon` if :math:`N\geq \mathcal{O}\left(\log{M} \max_i ||O_i||^2_{\text{shadow}}/\epsilon^2\right)`
# The shadow norm :math:`||O_i||^2_{\text{shadow}}` again depends on the unitary ensemble that is chosen.
#
# In [Huang2020]_, two different ensembles are considered:
#
# 1. Random :math:`n`-qubit Clifford circuits.
# 2. Tensor products of random single-qubit Clifford circuits.
#
# Although ensemble 1. leads to the most powerful estimators, it comes with serious practical limitations
# since :math:`n^2 / \log(n)` entangling gates are required to sample the Clifford circuit.
# For the purposes of this demo, we therefore choose ensemble 2., which is a more NISQ friendly approach.
#
# This ensemble comes with a significant drawback: The shadow norm :math:`||O_i||^2_{\text{shadow}}`
# becomes dependent on the locality :math:`k` of the observables that we want to estimate
#
# .. math::
#
#      ||O_i||^2_{\text{shadow}} \leq 4^k ||O_i||_\infty^2.
#
# This is a serious limitation. Say that we want to estimate the single Pauli observable
# :math:`\langle X_1 \otimes X_2 \otimes \ldots \otimes X_n \rangle`. Estimating this from repeated measurements
# would require :math:`1/\epsilon^2` samples, whereas we would need an exponentially large shadow due to :math:`k=n`.
# Therefore, classical shadows based on Pauli measurements offer only an advantage when we have to measure a large number
# of non-commuting observables with modest locality reduced locality.
#
# To perform classical shadow estimation, we require a couple of functions.
# Creating a shadow of size :math:`N` requires the following steps:
#
# 1. A quantum state :math:`\rho` is prepared.
# 2. A randomly selected unitary :math:`U` is applied
# 3. A computational basis measurement is performed.
# 4. The process is repeated :math:`N` times.
#
# First create the ``./classical_shadows.py`` file and import the following libraries.

# ./classical_shadows.py
import pennylane as qml
import pennylane.numpy as np
from typing import List

##############################################################################
# Then, create the following function, which takes as input a ``circuit_template`` with ``params``
# and returns a shadow of size ``shadow_size`` on ``num_qubits`` qubits.

def calculate_classical_shadow(
    circuit_template, params, shadow_size: int,num_qubits: int
) -> np.ndarray:
    """
    Given a circuit, creates a collection of snapshots U^dag|b><b| U with the stabilizer
    description.

    Args:
        circuit_template: A Pennylane QNode.
        params: Circuit parameters.
        shadow_size: The number of snapshots in the shadow.
        num_qubits: The number of qubits in the circuit.

    Returns:
        Numpy array containing the outcomes (0, 1) in the first `num_qubits` columns and
        the sampled Pauli's
        (0,1,2=x,y,z) in the final `num_qubits` columns.
    """
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]
    # sample random pauli unitaries uniformly, where 0,1,2 = X,Y,Z
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
# Furthermore, we'll need to import the functions from the source code::
#
#     # ./test_classical_shadows.py
#
#     from classical_shadows import calculate_classical_shadow
#
# Now we're ready to start writing tests. Note that ``calculate_classical_shadow`` only
# works if make sure that ``circuit_template`` returns only a shot. To this end, we create
# a PyTest fixture for a circuit that can be reused across multiple tests.

# ./test_classical_shadows.py

@pytest.fixture
def circuit_1_observable(request):
    """Circuit with single layer requiring nqubits parameters"""
    num_qubits = request.param
    # Make sure that the number of shots is set to 1
    dev = qml.device('default.qubit', wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, wires, **kwargs):
        observables = kwargs.pop('observable')
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return [qml.expval(o) for o in observables]
    # Return the shape of the parameters so other tests know how to initialize the
    # parameters. None = num_qubits
    param_shape = (None,)
    return circuit, param_shape, num_qubits

#########################################
# Using this fixture, we can write a test to verify that the output of ``calculate_classical_shadows``
# is correct. We expect the outcomes to be :math:`(+1,-1)` on each qubit, corresponding to the computational
# basis states :math:`|0\rangle, |1\rangle`. Furthermore, we expect the applied Pauli measurements
# to be returned as an integer :math:`(X,Y,Z)\to(0,1,2)` on each qubit. By adding the
# ``@pytest.mark.parametrize`` decorator, we can run this test multiple times, for different numbers of qubits
# Here, we run ``calculate_classical_shadow`` for :math:`n=1,2,3,4` qubits and verify that
# the returned output is correct.

# ./test_classical_shadows.py

@pytest.mark.parametrize("circuit_1_observable", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_1(circuit_1_observable, shadow_size=10):
    """Test calculating the shadow for a simple circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    outcomes = calculate_classical_shadow(circuit_template, params, shadow_size,
                                          num_qubits)
    assert all(o in [1.0, -1.0] for o in np.unique(outcomes[:, :num_qubits]))
    assert all(o in list(range(3)) for o in np.unique(outcomes[:, num_qubits:]))

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


# construct circuit 1 with a different number of qubits.
@pytest.mark.parametrize("circuit_2_observable", [1, 2, 3, 4], indirect=True)
def test_calculate_classical_shadow_circuit_2(circuit_2_observable, shadow_size=10):
    """Test calculating the shadow for a circuit with multiple layers"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    assert all(o in [1.0, -1.0] for o in np.unique(shadow[:, :num_qubits]))

#########################################
# If we now run ``pytest test_classical_shadows.py``, we see that the test passes,
#
# Spoof pytest output


@pytest.mark.parametrize("circuit_1_observable, shadow_size",
                         [[2, 10], [2, 100], [2, 1000], [2, 10000]],
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
# State Reconstruction from a Classical Shadow
# ############################################
#
# To verify that the classical shadow approximates the exact state that we want to estimate,
# we tomographically reconstruct the original quantum state :math:`\rho` from a classical
# shadow obtained from :math:`\rho`. Remember that we can approximate the :math:`\rho` by averaging
# over the snapshots and applying the inverse measurement channel
#
# .. math::
#
#     \rho = \mathbb{E}\left[\mathcal{M}^{-1}(U^{\dagger}|\hat{b}\rangle\langle\hat{b}|U)\right]
#
# The expectation :math:`\mathbb{E}[\cdot]` simply describes the average over the measurements
# :math:`|b\rangle` and the sampled unitaries.
# Inverting the measurment channel may seem formidable at first, however, Huang et al. [Huang2020]_
# show that for Pauli measurements we end up with a rather convenient expression,
#
# .. math::
#
#     \rho=\mathbb{E}[\hat{\rho}], \quad \text{where} \quad
#     \hat{\rho} = \bigotimes_{j=1}^n(3U^{\dagger}_j|\hat{b}_j\rangle\langle\hat{b}_j|U_j-\mathbb{I}).
#
# Here :math:`\hat{\rho}` is a snapshot state reconstructed from a single sample in the
# classical shadow, and :math:`\rho` is the average over all snapshot states :math:`\hat{\rho}` in the
# shadow.
#
# To implement the state reconstruction of :math:`\rho` in PennyLane, we develop two function
# ``snapshot_state`` and ``shadow_state_reconstruction``.
# To begin, we define the ``snapshot_state`` function that applies the above expression
# for :math:`\hat{\rho}`.
# Now, add the following code to the ``./classical_shadows.py`` file.

# ./classical_shadows.py

def snapshot_state(b_list, obs_list):
    """
    Reconstruct a state from a single snapshot in a shadow.

    **Details:**
    Implements Eq. (S44) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        b_list (array): The list of classical outcomes for the snapshot.
        obs_list (array): Indices for the applied pauli measurement
    """
    num_qubits = len(b_list)

    # computational basis states
    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    # local qubit unitaries
    phase_z = np.array([[1, 0],[0, -1j]], dtype=np.complex)
    hadamard = qml.Hadamard(0).matrix
    identity = qml.Identity(0).matrix

    # rotations from computational basis to x,y,z respectively
    unitaries = [hadamard, hadamard @ phase_z, identity]

    # reconstructing the snapshot state from local pauli measurements
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        # applying Eq. (S44)
        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


##############################################################################
# Next, we will write the ``shadow_state_reconstruction`` method to construct a
# shadow state approximation for :math:`\rho`.

# ./classical_shadows.py
def shadow_state_reconstruction(shadow):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.

    Args:
        shadow (array): a shadow matrix obtained from `calculate_classical_shadow`.
    """
    num_snapshots = shadow.shape[0]
    num_qubits = shadow.shape[1] // 2

    # classical values
    b_lists = shadow[:, 0:num_qubits]
    # pauli observable ids
    obs_lists = shadow[:, num_qubits:2 * num_qubits]

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=np.complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots

##############################################################################
# Now, we will test the ``shadow_state_reconstruction`` function.
# First we will write a unit test named ``test_shadow_state_reconstruction_unit``.
# As input, this test will take a classical shadow and matrix representing the
# expected state reconstruction.
# This test function will be used to verify three key features of ``shadow_state_reconstruction``.
#
# 1. The application of local unitary :math:`U` used to reconstruct a single qubit snapshot.
# 2. The tensor product structure for multi-qubit reconstructions.
# 3. The averaging over multiple snapshots.

# ./test_classical_shadows.py

# 1. Verify that each possible qubit snapshot reconstructs the correct
# snapshot state.
qubit_snapshot_test_cases = [
    (np.array([[1,0]]), np.array([[0.5,1.5],[1.5,0.5]])),
    (np.array([[-1,0]]), np.array([[0.5,-1.5],[-1.5,0.5]])),
    (np.array([[1,1]]), np.array([[0.5,-1.5j],[1.5j,0.5]])),
    (np.array([[-1,1]]), np.array([[0.5,1.5j],[-1.5j,0.5]])),
    (np.array([[1,2]]), np.array([[2,0],[0,-1]])),
    (np.array([[-1,2]]), np.array([[-1,0],[0,2]]))
]

# 2. Verify that two-qubit states are tensored together correctly.
two_qubit_snapshot_test_cases = [
    (np.array([[1,1,2,2]]), np.array([[2,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])),
    (np.array([[1,-1,2,2]]), np.array([[-1,0,0,0],[0,2,0,0],[0,0,-1,0],[0,0,0,1]])),
]

# 3. Verify that the snapshots in each shadow are averaged together.
qubit_shadow_test_cases = [
    (np.array([[1,0],[-1,0],[1,2]]), np.array([[1,0],[0,0]])),
    (np.array([[1,0],[1,2],[-1,2],[1,1],[-1,1]]), np.array([[0.5,0.5],[0.5,0.5]])),
    (np.array([[1,0],[-1,0],[1,1],[-1,1],[1,2],[-1,2]]), np.array([[0.5,0],[0,0.5]]))
]

@pytest.mark.parametrize(
    "shadow,reconstructed_state_match",
    qubit_snapshot_test_cases + two_qubit_snapshot_test_cases + qubit_shadow_test_cases
)
def test_shadow_state_reconstruction_unit(shadow, reconstructed_state_match):
    reconstructed_state = shadow_state_reconstruction(shadow)

    # reconstructed shadow states are trace-one
    assert np.isclose(np.trace(reconstructed_state), 1, atol=1e-7)
    assert reconstructed_state.all() == reconstructed_state_match.all()


##############################################################################
# At this point, we've verified that ``shadow_state_reconstruction`` behaves
# as expected.
# The next step is to test its integration with the ``calculate_classical_shadow``
# function.
# To do this, we will write the ``test_shadow_state_reconstruction_integration``
# function which makes use of the test fixtures ``circuit_1_observable`` and
# ``circuit_1_state`` to setup devices for out tests.

# ./test_classical_shadows.py

# test fixture that obtains the state vector for the preparation circuit.
# We'll use this as the target state for reconstruction.
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

@pytest.mark.parametrize(
    "circuit_1_observable, circuit_1_state, params, shadow_size",
    [[2, 2, [0,0], 1000], [3, 3, [np.pi/4,np.pi/3,np.pi/2], 1000]],
    indirect=['circuit_1_observable','circuit_1_state']
)
def test_shadow_state_reconstruction_integration(circuit_1_observable, circuit_1_state, params, shadow_size):
    """Test reconstructing the state from the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    circuit_template_state, _, _ = circuit_1_state

    # calculating shadow and reconstructing state
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)
    shadow_state = shadow_state_reconstruction(shadow)

    # calculating exact quantum state
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    exact_state_ket = circuit_template_state(
        params, wires=dev_exact.wires, observable=[qml.state()]
    )
    exact_state = np.outer(exact_state_ket, exact_state_ket.conj())

    # verifying the reconstructed shadow state has trace-one
    assert np.isclose(np.trace(shadow_state), 1, atol=1e-7)
    # verifying that the shadow state roughly approximates the exact state
    assert np.isclose(shadow_state.all(), exact_state.all(), atol=0.1)

##############################################################################
# Example: Reconstructing a Bell State
# ************************************
#
# First, we construct a single-shot, ``'default.qubit'`` device and
# define the ``bell_state_circuit`` qnode to construct and measure a Bell state.

# ./notebook_classical_shadows.ipynb
num_qubits = 2

dev = qml.device('default.qubit', wires=num_qubits, shots=1)

# circuit to create a Bell state and measure it in
# the bases specified by the 'observable' keyword warg.
@qml.qnode(device=dev)
def bell_state_circuit(params, wires, **kwargs):
    observables = kwargs.pop('observable')

    qml.Hadamard(0)
    qml.CNOT(wires=[0,1])

    return [qml.expval(o) for o in observables]

##############################################################################
# Then, we'll construct a classical shadow consisting of 1000 snapshots.

# ./notebook_classical_shadows.ipynb
num_snapshots = 1000
params = []

shadow = calculate_classical_shadow(bell_state_circuit, params, num_snapshots, num_qubits)
shadow


##############################################################################
# The bell state is then reconstructed with ``shadow_state_reconstruction``.

# ./notebook_classical_shadows.ipynb
shadow_state = shadow_state_reconstruction(shadow)
shadow_state

##############################################################################
# Note the resemblance to the exact Bell state density matrix.

# ./notebook_classical_shadows.ipynb
bell_state = np.array([
   [0.5, 0, 0, 0.5],
   [  0, 0, 0,   0],
   [  0, 0, 0,   0],
   [0.5, 0, 0, 0.5]
])

##############################################################################
# To measure the closeness we can use the operator norm.

# ./classical_shadows.py
def operator_2_norm(R):
    """
    Calculate the operator two norm.
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))

# calculating trace-distance between ideal bell state and shadow reconstruction
operator_2_norm(bell_state - shadow_state)

##############################################################################
# Finally, we see how the approximation improves as we increase the
# number of snapshots.

# ./notebook_classical_shadows.ipynb
import matplotlib.pyplot as plt

trace_distances = []
snapshots_range = range(100,10000,1000)
for num_snapshots in snapshots_range:
    shadow = calculate_classical_shadow(bell_state_circuit, params, num_snapshots, num_qubits)
    shadow_state = shadow_state_reconstruction(shadow)

    tr_distance = np.real(operator_2_norm(bell_state - shadow_state))
    trace_distances.append(tr_distance)

plt.plot(snapshots_range, trace_distances)
plt.title("Trace Distance between Ideal and Shadow Bell States")
plt.xlabel("Number of Snapshots")
plt.ylabel("Trace Distance")
plt.show()

##############################################################################
# As expected, when the number of snapshots increases, the state reconstruction
# becomes closer to the ideal state.


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
    Calculate the estimator E[O] = sum_i Tr{rho_i O} where rho_i is a snapshot in the
    shadow.
    
    Args:
        shadows: Numpy array containing the outcomes (0, 1) in the first `num_qubits`
        columns and the sampled Pauli's (0,1,2=x,y,z) in the final `num_qubits` columns.
        observable: Single PennyLane observable consisitng of single Pauli operators e.g.
        qml.PauliX(0) @ qml.PauliY(1)
    
    Returns:
        Scalar corresponding to the estimate of the observable.
    """
    map_name_to_int = {'PauliX': 0, 'PauliY': 1, 'PauliZ': 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        observable_as_list = [(map_name_to_int[observable.name], observable.wires[0])]
    else:
        observable_as_list = [(map_name_to_int[o.name], o.wires[0]) for o in
                              observable.obs]

    num_qubits = shadows.shape[1] // 2
    sum_product, cnt_match = 0, 0
    # loop over the shadows:
    for single_measurement in shadows:
        not_match = 0
        product = 1
        # loop over all the paulis that we care about
        for pauli_XYZ, position in observable_as_list:
            # if the pauli in our shadow does not match, we break and go to the next
            # shadow
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


def shadow_bound(M: int, error: float, max_k: int, observables: List[np.ndarray]):
    """
    Calculate the shadow bound for the pauli measurement scheme.
    """
    shadow_norm = lambda op: np.linalg.norm(op, ord=np.inf) ** 2
    return int(np.ceil(
        np.log(M) * 4 ** max_k * max(shadow_norm(o) for o in observables) / error ** 2))


##############################################################################
# Testing ``estimate_shadow_observable``

# ./test_classical_shadows.py

# TODO: create a fixture for the shadow so we only have run it once
@pytest.mark.parametrize("circuit_1_observable, shadow_size",
                         [[2, 10], [2, 100], [2, 1000], [2, 10000]],
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
    """Test calculating multiple observables with the shadowsize shadow_size given by the
     bound in the paper"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    list_of_observables = [qml.PauliX(0) @ qml.PauliZ(2) @ qml.PauliY(4),
                           qml.PauliY(1) @ qml.PauliZ(2),
                           qml.PauliX(0),
                           qml.PauliY(3) @ qml.PauliZ(4) @ qml.PauliY(num_qubits - 1)]
    # Calculate how many shadows we need to get an error of 1e-1 for this set of
    # observables
    shadow_size = shadow_bound(M=len(list_of_observables), max_k=3,
                               error=1e-1,
                               observables=[o.matrix for o in list_of_observables])

    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    expval_shadow = sum(estimate_shadow_obervable(shadow, o) for o in list_of_observables)
    dev_exact = qml.device('default.qubit', wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = sum(
        circuit_template(params, wires=dev_exact.wires, observable=[o, ]) for o in
        list_of_observables)
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")
    # from the theoretical bound we know that the variance of this estimator must be
    # within 1e-1
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
# .. [Huang2020] Huang, Hsin-Yuan, Richard Kueng, and John Preskill.
#             "Predicting many properties of a quantum system from very few measurements."
#             Nature Physics 16.10 (2020): 1050-1057.
