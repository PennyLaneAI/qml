r"""
Classical Shadows
=================
.. meta::
    :property="og:description": Learn how to construct classical shadows
        and use them to estimate observables.
    :property="og:image": https://pennylane.ai/qml/_images/atom_shadow.png

.. related::

*Authors: Roeland Wiersema & Brian Doolittle (Xanadu Residents).
Posted: 7 May 2021. Last updated: 7 May 2021.*

Estimating properties of unknown quantum states is a key objective of quantum
information science and technology.
For example, one might want to check whether an apparatus prepares a particular target state,
or verify that an unknown system can prepare entangled states.
In principle, any unknown quantum state can be fully characterized by performing `quantum state
tomography <http://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf>`_.
However this procedure requires one to acquire accurate expectation values for a set of observables
which grows exponentially with the number of qubits.
A potential workaround for these scaling concerns is provided by the classical shadow approximation
introduced in `Predicting many properties of a quantum system from very few measurements
<https://arxiv.org/pdf/2002.08953.pdf>`_ [[#Huang2020]_].

The classical shadow approximation is an efficient protocol for constructing a *classical shadow*
representation of an unknown quantum state.
The classical shadow can be used to estimate properties such as
quantum state fidelity, Hamiltonian observables, and two-point correlators.

.. figure:: ../demonstrations/classical_shadows/classical_shadow_overview.png
    :align: center
    :width: 80%

    (Image from Huang et al. [[#Huang2020]_].)

In this demo, we will use PennyLane to construct classical shadows and use them to reconstruct
quantum states and estimate observables.
We will use a test oriented approach where we develop a ``classical_shadows.py`` library complete
with tests and Jupyter notebook examples.
We will work in three separate files:

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
# we have an information-theoretic bound that controls the precision of our estimator.
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
# as a post-processing step to calculate some function of the state.
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
# In fact, the authors of [[#Huang2020]_] prove that with a shadow of size :math:`N`, we can predict :math:`M` arbitary linear functions
# :math:`\text{Tr}{O_1\rho},\ldots,\text{Tr}{O_M \rho}` to additive error :math:`\epsilon` if :math:`N\geq \mathcal{O}\left(\log{M} \max_i ||O_i||^2_{\text{shadow}}/\epsilon^2\right)`.
# The shadow norm :math:`||O_i||^2_{\text{shadow}}` again depends on the unitary ensemble that is chosen.
#
# Two different ensembles are considered:
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
# would require :math:`1/\epsilon^2` samples, whereas we would need an exponentially large shadow due to the :math:`4^n` appearing in the bound.
# Therefore, classical shadows based on Pauli measurements offer only an advantage when we have to measure a large number
# of non-commuting observables with modest locality.
#
# To perform classical shadow estimation, we require a couple of functions.
# We will write the core functions in ``./classical_shadows.py`` file and import the following libraries:

# ./classical_shadows.py
import pennylane as qml
import pennylane.numpy as np
from typing import List

np.random.seed(666)

##############################################################################
# Creating a shadow of size :math:`N` requires the following steps:
#
# 1. A quantum state :math:`\rho` is prepared.
# 2. A randomly selected unitary :math:`U` is applied
# 3. A computational basis measurement is performed.
# 4. The process is repeated :math:`N` times.
# With this in mind, add the ``calculate_classical_shadow`` function below.
# This function obtains a classical shadow for the state prepared by the
# ``circuit_template``, where the classical shadow is simply represented by
# a matrix where each row is a distinct snapshot.


def calculate_classical_shadow(
    circuit_template, params, shadow_size: int, num_qubits: int
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
# ``test_classical_shadows.py`` and import the following libraries:

# ./test_classical_shadows.py

import pytest
import pennylane as qml
import pennylane.numpy as np

np.random.seed(666)
##############################################################################
# The package `pytest <https://docs.pytest.org/en/6.2.x/>`_ is a code testing framework that makes
# it easy to test complex code bases. This is ideal of you are working on a project with
# a lot of moving parts that will be developed over a longer period of time. By
# implementing automated testing, developers can make sure that their codebase is
# functioning as expected. This prevents bugs from being introduced as a result of changes
# to the code.
#
# In addition to these Python libraries, we will need to import the functions from
# the source code
#
# .. code-block:: python
#
#     # ./test_classical_shadows.py
#
#     from classical_shadows import calculate_classical_shadow
#
# Now we are ready to start writing tests. Note that ``calculate_classical_shadow`` only
# works if we make sure that ``circuit_template`` returns only a shot. To this end, we create
# a PyTest fixture for a circuit that can be reused across multiple tests. This speeds up
# testing since we do not have to recreate the circuits multiple time across many different
# tests.

# ./test_classical_shadows.py


@pytest.fixture
def circuit_1_observable(request):
    """Circuit with single layer requiring nqubits parameters"""
    num_qubits = request.param
    # Make sure that the number of shots is set to 1
    dev = qml.device("default.qubit", wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, wires, **kwargs):
        observables = kwargs.pop("observable")
        for w in dev.wires:
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        return [qml.expval(o) for o in observables]

    # Return the shape of the parameters so other tests know how to initialize the
    # parameters. None = num_qubits
    param_shape = (None,)
    return circuit, param_shape, num_qubits


##############################################################################
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
    # initialize the parameters from the returned shape
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    outcomes = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)
    assert all(o in [1.0, -1.0] for o in np.unique(outcomes[:, :num_qubits]))
    assert all(o in list(range(3)) for o in np.unique(outcomes[:, num_qubits:]))


@pytest.fixture
def circuit_2_observable(request):
    """Circuit with multiple layers requiring nqubits*3 parameters"""
    num_qubits = request.param
    dev = qml.device("default.qubit", wires=num_qubits, shots=1)

    @qml.qnode(device=dev)
    def circuit(params, wires, **kwargs):
        observables = kwargs.pop("observable")
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
    # initialize the parameters from the returned shape
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    assert all(o in [1.0, -1.0] for o in np.unique(shadow[:, :num_qubits]))


##############################################################################
# If we now run ``pytest test_classical_shadows.py``, we see that the tests pass.
#
# .. code-block::
#
#     $ pytest test_classical_shadows.py
#     ============================= test session starts =============================
#     platform darwin -- Python 3.7.4, pytest-6.2.4, py-1.10.0, pluggy-0.13.0
#     rootdir: /path/to/working/directory
#     plugins: arraydiff-0.3, remotedata-0.3.2, doctestplus-0.4.0, openfiles-0.4.0
#     collected 8 items
#
#     test_classical_shadows.py ........                                       [100%]
#
#     =============================== 8 passed in 5.43s =============================
#
# Finally, we will show how to apply the ``calculate_classical_shadow`` function.
# Launch a Jupyter notebook server and create a notebook called
# ``notebook_classical_shadows.ipynb``.

# ./notebook_classical_shadows.ipynb
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import time

##############################################################################
# Furthermore, we will want to add the ``calculate_classical_shadow`` method.
#
# .. code-block:: python
#
#     # ./notebook_classical_shadows.ipynb
#     from classical_shadows import calculate_classical_shadow
#
# Now we will check the performance of ``calculate_classical_shadow``.
# First, we will create a two-qubit device and circuit that applies an
# ``RY`` rotation to each qubit.

# ./notebook_classical_shadows.ipynb
num_qubits = 2

# setting up a two-qubit circuit
dev = qml.device("default.qubit", wires=num_qubits, shots=1)


@qml.qnode(device=dev)
def local_qubit_rotation_circuit(params, wires, **kwargs):
    observables = kwargs.pop("observable")
    for w in dev.wires:
        qml.RY(params[w], wires=w)

    return [qml.expval(o) for o in observables]


# arrays in which to collect data
elapsed_times = []
shadows = []

# collecting shadows and elapsed times
params = np.random.randn(2)
for num_snapshots in [10, 100, 1000, 10000]:
    start = time.time()
    shadow = calculate_classical_shadow(
        local_qubit_rotation_circuit, params, num_snapshots, num_qubits
    )
    elapsed_times.append(time.time() - start)
    shadows.append(shadow)

# printing out the smallest shadow as an example
shadows[0]

##############################################################################
# Observe that the shadow is simply a matrix.
# For two qubits, the first two columns describe the outcome for the pauli observable
# while the second two columns index the pauli observable for each qubit.
# We now plot the computation times taken to acquire the shadows as the number of
# snapshots increases.

# ./notebook_classical_shadows.ipynb
plt.plot([10, 100, 1000, 10000], elapsed_times)
plt.title("Time taken to obtain a classical shadow from a 2-qubit state")
plt.xlabel("Number of Snapshots in Shadow")
plt.ylabel("Elapsed Time")
plt.show()

##############################################################################
# As one might expect, the computation time increases linearly with the number
# of snapshots.

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
# Inverting the measurment channel may seem formidable at first, however, Huang et al.
# [[#Huang2020]_]
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
# To implement the state reconstruction of :math:`\rho` in PennyLane, we develop the
# ``shadow_state_reconstruction`` function.
# Add the following code to the ``./classical_shadows.py`` file.

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
    obs_lists = shadow[:, num_qubits : 2 * num_qubits]

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots


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
    phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
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
# Now, we will test the ``shadow_state_reconstruction`` function.
# The first thing to do is import the function into the test file:
#
# .. code-block:: python
#
#     # ./test_classical_shadows.py
#     from classical_shadows import shadow_state_reconstruction
#
# Then, we will write a unit test named ``test_shadow_state_reconstruction_unit``.
# As input, this test will take a classical shadow and matrix representing the
# expected state reconstruction.
# This test function will be used to verify three key features of ``shadow_state_reconstruction``.
#
# 1. The application of local unitary :math:`U` used to reconstruct a single qubit snapshot.
# 2. The tensor product structure for multi-qubit reconstructions.
# 3. The averaging over multiple snapshots.
#
# The code for this test test is exemplified below.

# ./test_classical_shadows.py

# 1. Verify qubit snapshot state reconstruction.
qubit_snapshot_test_cases = [
    (np.array([[1, 0]]), np.array([[0.5, 1.5], [1.5, 0.5]])),
    (np.array([[-1, 0]]), np.array([[0.5, -1.5], [-1.5, 0.5]])),
    (np.array([[1, 1]]), np.array([[0.5, -1.5j], [1.5j, 0.5]])),
    (np.array([[-1, 1]]), np.array([[0.5, 1.5j], [-1.5j, 0.5]])),
    (np.array([[1, 2]]), np.array([[2, 0], [0, -1]])),
    (np.array([[-1, 2]]), np.array([[-1, 0], [0, 2]])),
]

# 2. Verify that two-qubit states are tensored together correctly.
two_qubit_snapshot_test_cases = [
    (
        np.array([[1, 1, 2, 2]]),
        np.array([[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ),
    (
        np.array([[1, -1, 2, 2]]),
        np.array([[-1, 0, 0, 0], [0, 2, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ),
]

# 3. Verify that the snapshots in each shadow are averaged together.
qubit_shadow_test_cases = [
    (np.array([[1, 0], [-1, 0], [1, 2]]), np.array([[1, 0], [0, 0]])),
    (np.array([[1, 0], [1, 2], [-1, 2], [1, 1], [-1, 1]]), np.array([[0.5, 0.5], [0.5, 0.5]])),
    (np.array([[1, 0], [-1, 0], [1, 1], [-1, 1], [1, 2], [-1, 2]]), np.array([[0.5, 0], [0, 0.5]])),
]


@pytest.mark.parametrize(
    "shadow,reconstructed_state_match",
    qubit_snapshot_test_cases + two_qubit_snapshot_test_cases + qubit_shadow_test_cases,
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
# We will use this as the target state for reconstruction.
@pytest.fixture
def circuit_1_state(request):
    """Circuit with single layer requiring num_qubits parameters"""
    num_qubits = request.param
    dev = qml.device("default.qubit", wires=num_qubits)

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
    [[2, 2, [0, 0], 1000], [3, 3, [np.pi / 4, np.pi / 3, np.pi / 2], 1000]],
    indirect=["circuit_1_observable", "circuit_1_state"],
)
def test_shadow_state_reconstruction_integration(
    circuit_1_observable, circuit_1_state, params, shadow_size
):
    """Test reconstructing the state from the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    circuit_template_state, _, _ = circuit_1_state

    # calculating shadow and reconstructing state
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)
    shadow_state = shadow_state_reconstruction(shadow)

    # calculating exact quantum state
    dev_exact = qml.device("default.qubit", wires=num_qubits)
    exact_state_ket = circuit_template_state(
        params, wires=dev_exact.wires, observable=[qml.state()]
    )
    exact_state = np.outer(exact_state_ket, exact_state_ket.conj())

    # verifying the reconstructed shadow state has trace-one
    assert np.isclose(np.trace(shadow_state), 1, atol=1e-7)
    # verifying that the shadow state roughly approximates the exact state
    assert np.isclose(shadow_state.all(), exact_state.all(), atol=0.1)


##############################################################################
# To run the tests, simply use the command ``$ pytest ./test_classical_shadows.py``.
# The output of the tests should look something like this:
#
# .. code-block::
#
#     $ pytest test_classical_shadows.py
#     ============================= test session starts =============================
#     platform darwin -- Python 3.7.4, pytest-6.2.4, py-1.10.0, pluggy-0.13.0
#     rootdir: /path/to/working/directory
#     plugins: arraydiff-0.3, remotedata-0.3.2, doctestplus-0.4.0, openfiles-0.4.0
#     collected 21 items
#
#     test_classical_shadows.py ........                                       [100%]
#
#     =================== 21 passed, 1 warning in 237.25s (0:03:57) ==================
#
# Example: Reconstructing a Bell State
# ************************************
#
# First, we will need to import the ``shadow_state_reconstruction`` method.
#
# .. code-block:: python
#
#     # ./notebook_classical_shadows.ipynb
#     from classical_shadows import shadow_state_reconstruction
#
# Note that you may need to restart the notebook kernel in order to import the new method.
# Then, we construct a single-shot, ``'default.qubit'`` device and
# define the ``bell_state_circuit`` QNode to construct and measure a Bell state.

# ./notebook_classical_shadows.ipynb

num_qubits = 2

dev = qml.device("default.qubit", wires=num_qubits, shots=1)

# circuit to create a Bell state and measure it in
# the bases specified by the 'observable' keyword argument.
@qml.qnode(device=dev)
def bell_state_circuit(params, wires, **kwargs):
    observables = kwargs.pop("observable")

    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])

    return [qml.expval(o) for o in observables]


##############################################################################
# Then, we will construct a classical shadow consisting of 1000 snapshots.

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
bell_state = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])

##############################################################################
# To measure the closeness we can use the operator norm.
# We will add this utility method to the notebook for convenience.

# ./notebook_classical_shadows.ipynb
def operator_2_norm(R):
    """
    Calculate the operator two norm.
    """
    return np.sqrt(np.trace(R.conjugate().transpose() @ R))


# Calculating the distance between ideal and shadow states.
operator_2_norm(bell_state - shadow_state)

##############################################################################
# Finally, we see how the approximation improves as we increase the
# number of snapshots.

# ./notebook_classical_shadows.ipynb
trace_distances = []
snapshots_range = range(100, 1000, 10000)
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
# Estimating Pauli Observables with Classical Shadows
# ###################################################
#
# We have confirmed that our code produces classical shadows that can be used to reconstruct
# the state. However, the goal of classical shadows is not to perform full tomography, which takes
# an exponential amount of resources. Instead, we want to use the shadows to efficiently
# calculate linear functions of a quantum state. To do this, we write a function
# ``estimate_shadow_observable`` that takes in the previously constructed shadow
# :math:`S(\rho, N)=[\hat{\rho}_1,\hat{\rho}_1,\ldots,\hat{\rho}_N]`, and
# estimates any observable via a median of means estimation. This makes the estimator
# more robust to outliers and is required to formally prove the aforementioned theoretical
# bound. Also, the bound has a failure probability :math:`\delta`, and by choosing :math:`K` to be
# suitably large, we can exponentially surpress this failure probability.
# The procedure is simple: split up the shadow into :math:`K` equally sized chunks
# and estimate the mean for each of these chunks,
#
# .. math::
#
#      \langle O_{(k)}\rangle &= \text{Tr}\{O \hat{\rho}_{(k)}\}\\
#      \hat{\rho}_{(k)} &= \frac{1}{ \lfloor N/K \rfloor } \sum_{i=(k-1)\lfloor N/K \rfloor + 1}^{k \lfloor N/K \rfloor } \hat{\rho}_i.
#
# The median of means estimator is then simply the median of this set
#
# .. math::
#
#       \langle O\rangle &= \text{median}\{\langle O_{(1)} \rangle,\ldots, \langle O_{(K)} \rangle \}.
#
# Assume now that :math:`O=\bigotimes_j^n P_j`, where :math:`P_j \in \{I, X, Y, Z\}`.
# To efficiently calculate the estimator for :math:`O`, we look at a single snapshot outcome and plug in the inverse measurement channel:
#
# .. math::
#
#    \text{Tr}\{O\hat{\rho}_i\} &= \text{Tr}\{\bigotimes_{j=1}^n P_j (3U^{\dagger}_j|\hat{b}_j\rangle\langle\hat{b}_j|U_j-\mathbb{I})\}\\
#     &= \prod_j^n \text{Tr}\{ 3 P_j U^{\dagger}_j|\hat{b}_j\rangle\langle\hat{b}_j|U_j\}.
#
# Due to the orthogonality of the Pauli operators, this evaluates to :math:`\pm 3` if :math:`P_j` the
# corresponding measurement basis :math:`U_j` and is 0 otherwise. Hence if a single :math:`U_j` in the snapshot
# does not match the one in :math:`O`, the whole product evaluates to zero. As a result, calculating the mean estimator
# can be reduced to counting the number of exact matches in the shadow with the observable, and multiplying with the appropriate
# sign.

# ./classical_shadows.py


def estimate_shadow_obervable(shadows, observable, k=10) -> float:
    """
    Adapted from https://github.com/momohuang/predicting-quantum-properties
    Calculate the estimator E[O] = median(Tr{rho_{(k)} O}) where rho_(k)) is set of k
    snapshots in the shadow.

    Args:
        shadows: Numpy array containing the outcomes (0, 1) in the first `num_qubits`.
        columns and the sampled Pauli's (0,1,2=x,y,z) in the final `num_qubits` columns.
        observable: Single PennyLane observable consisting of single Pauli operators e.g.
        qml.PauliX(0) @ qml.PauliY(1).
        k: number of chunks in the median of means estimator.

    Returns:
        Scalar corresponding to the estimate of the observable.
    """
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        observable_as_list = [(map_name_to_int[observable.name], observable.wires[0])]
    else:
        observable_as_list = [(map_name_to_int[o.name], o.wires[0]) for o in observable.obs]

    num_qubits = shadows.shape[1] // 2
    shadow_size = shadows.shape[0]
    sum_product, cnt_match = 0, 0
    means = []
    for i in range(0, shadow_size, shadow_size // k):

        # loop over the shadows:
        for single_measurement in shadows[i : i + shadow_size // k]:
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
            # do not record the shadow
            if not_match == 1:
                continue

            sum_product += product
            cnt_match += 1
    if cnt_match == 0:
        means.append(0)
    else:
        means.append(sum_product / cnt_match)
    return np.median(means)


##############################################################################
# Next, we can define a function that calculates the number of samples
# required to get an error :math:`\epsilon` on our estimator for a given set of observables.

# ./classical_shadows.py
def shadow_bound(error: float, max_k: int, observables: List[np.ndarray]) -> int:
    """
    Calculate the shadow bound for the pauli measurement scheme.

    Args:
        error: The error on the estimator.
        max_k: The maximum locality of all the observables.
        observables: List of matrices corresponding to the observables we intend to
        measure

    Returns:
        An integer that gives the number of samples required to satisfy the shadow bound.
    """
    M = len(observables)
    shadow_norm = lambda op: np.linalg.norm(op, ord=np.inf) ** 2
    return int(
        np.ceil(np.log(M) * (4 ** max_k) * max(shadow_norm(o) for o in observables) / error ** 2)
    )


##############################################################################
# We first test ``estimate_shadow_observable`` by estimating :math:`X_1 X_2` on the circuit.
# This test makes sure that given the expected input, the estimator returns a value in
# :math:`[-1, 1]`.

# ./test_classical_shadows.py


@pytest.mark.parametrize(
    "circuit_1_observable, shadow_size",
    [[2, 10], [2, 100], [2, 1000], [2, 10000]],
    indirect=["circuit_1_observable"],
)
def test_estimate_shadow_observable_single(circuit_1_observable, shadow_size):
    """Test calculating an observable with the shadow for a circuit with a single layer"""
    circuit_template, param_shape, num_qubits = circuit_1_observable
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)

    observable = [qml.PauliX(0) @ qml.PauliX(1)]
    expval_shadow = estimate_shadow_obervable(shadow, observable[0])
    assert -1.0 <= expval_shadow <= 1.0


##############################################################################
# We want to make sure that our code verifies the bound. To do this, we pick a
# set of :math:`M` observables and calculate with ``shadow_bound`` how many samples we
# need to get an estimator at most error :math:`\epsilon`. The test then calculates the
# exact expectation value and asserts if the classical shadow estimate is within :math:`\epsilon` of
# that value.

# ./test_classical_shadows.py


@pytest.mark.parametrize("circuit_2_observable", [8], indirect=["circuit_2_observable"])
def test_estimate_shadow_observable_shadow_bound(circuit_2_observable):
    """Test calculating multiple observables with the shadowsize shadow_size given by the
    bound in the paper"""
    circuit_template, param_shape, num_qubits = circuit_2_observable
    # initialize the parameters from the returned shape
    params = np.random.randn(*[s if (s != None) else num_qubits for s in param_shape])
    list_of_observables = [
        qml.PauliX(0) @ qml.PauliZ(2) @ qml.PauliY(4),
        qml.PauliY(1) @ qml.PauliZ(2),
        qml.PauliX(0),
        qml.PauliY(3) @ qml.PauliZ(4) @ qml.PauliY(num_qubits - 1),
    ]
    # Calculate how many shadows we need to get an error of 1e-1 for this set of
    # observables
    shadow_size = shadow_bound(
        max_k=3, error=1e-1, observables=[o.matrix for o in list_of_observables]
    )
    # get the shadow
    shadow = calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits)
    # estimate all observables, set partitioning to be high so failure probability is low.
    expval_shadow = sum(
        estimate_shadow_obervable(shadow, o, k=shadow_size // 10) for o in list_of_observables
    )
    # switch device for exact calculation
    dev_exact = qml.device("default.qubit", wires=num_qubits)
    # change the simulator to be the exact one.
    circuit_template.device = dev_exact
    expval_exact = sum(
        circuit_template(
            params,
            wires=dev_exact.wires,
            observable=[
                o,
            ],
        )
        for o in list_of_observables
    )
    print(f"Shadow : {expval_shadow} - Exact {expval_exact}")
    # from the theoretical bound we know that the variance of this estimator must be
    # within 1e-1
    assert np.isclose(expval_shadow, expval_exact, atol=1e-1)


##############################################################################
# If we again run ``$ pytest ./test_classical_shadows.py``, the output of the tests
# is
# .. code-block::
#
#     $ pytest test_classical_shadows.py
#     ============================= test session starts =============================
#     platform darwin -- Python 3.7.4, pytest-6.2.4, py-1.10.0, pluggy-0.13.0
#     rootdir: /path/to/working/directory
#     plugins: arraydiff-0.3, remotedata-0.3.2, doctestplus-0.4.0, openfiles-0.4.0
#     collected 26 items
#
#     test_classical_shadows.py ........                                       [100%]
#
#     =================== 26 passed, 1 warning in 237.25s (0:03:57) ==================


##############################################################################
# This confirms that our code is doing what we expect. Ideally, we want to add more
# tests and fixtures, so that we discover edge cases where things no longer work. However,
# for the purposes of this demo we will leave it like this.

##############################################################################
# Example: Estimating a simple set of observables
# *************************************************
# Here, we give an example for estimating multiple observables on a 10 qubit circuit.
# We first create a simple circuit

# ./notebook_classical_shadows.ipynb

num_qubits = 10
dev = qml.device("default.qubit", wires=num_qubits, shots=1)


@qml.qnode(device=dev)
def circuit(params, **kwargs):
    observables = kwargs.pop("observable")
    for w in range(num_qubits):
        qml.Hadamard(wires=w)
        qml.RY(params[w], wires=w)
    for w in dev.wires[:-1]:
        qml.CNOT(wires=[w, w + 1])
    for w in dev.wires:
        qml.RZ(params[w + num_qubits], wires=w)
    return [qml.expval(o) for o in observables]


params = np.random.randn(2 * num_qubits)
##############################################################################
# Next, we define our set of observables
#
# .. math::
#
#   O = \sum_i^{n-1} X_i X_{i+1}
#
#
list_of_observables = [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(num_qubits - 1)]
##############################################################################
# With the ``shadow_bound`` function, we can calculate how many shadows we need to get
# an error of 1e-1 for this set of observables

shadow_size_bound = shadow_bound(
    max_k=2, error=1e-1, observables=[o.matrix for o in list_of_observables]
)
shadow_size_bound
##############################################################################
# To see how the estimate improves, we consider different shadow sizes, up to :math:`10^4`
# snapshots.
shadow_size_grid = [10, 100, 1000, 5000, 10000]
estimates = []
for shadow_size in shadow_size_grid:
    shadow = calculate_classical_shadow(circuit, params, shadow_size, num_qubits)

    estimates.append(
        sum(estimate_shadow_obervable(shadow, o, k=shadow_size // 10) for o in list_of_observables)
    )
##############################################################################
# Then, we calculate the ground truth by changing the device backend
dev_exact = qml.device("default.qubit", wires=num_qubits)
# change the simulator to be the exact one.
circuit.device = dev_exact
expval_exact = sum(
    circuit(
        params,
        wires=dev_exact.wires,
        observable=[
            o,
        ],
    )
    for o in list_of_observables
)
##############################################################################
# If we plot the obtained estimates, we should see the error decrease as the number of
# snapshots increases (up to statistical fluctuations). Also, we plot the value of the
# bound.
plt.plot(shadow_size_grid, [np.abs(e - expval_exact) for e in estimates])
plt.scatter([shadow_size_bound], [1e-1], marker="*")
plt.show()
##############################################################################
# As expected, the bound is satisfied and the accuracy increases.

##############################################################################
# .. [#Huang2020] Huang, Hsin-Yuan, Richard Kueng, and John Preskill.
#             "Predicting many properties of a quantum system from very few measurements."
#             Nature Physics 16.10 (2020): 1050-1057.
