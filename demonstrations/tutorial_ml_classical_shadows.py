r"""
Learning ground state of 2D antiferromagnetic Heisenberg model
==============================================================

.. meta::
    :property="og:description": Learning ground state of 2-D
    antiferromagnetic Heisenberg model.
    :property="og:image": https://pennylane.ai/qml/_images/ml_classical_shadow.png

.. related::

   tutorial_classical_shadows Classical Shadows


*Author: Utkarsh. Posted: XX. Last Updated: XX March 2022*


In general, simulating an :math:`n`-qubit quantum mechanical system
becomes classically intractable as the number of classical bits required
to store the state of the system scale exponentially in :math:`n`. This
inhibits us from using classical simulation of larger quantum systems to
estimate their properties. In the past, this challenge has been
addressed by the quantum community using classical shadow formalism,
which allows one to build a concise classical description of the state
of a quantum system. Recently, it has been shown by Hsin-Yuan Huang et
al. [[#preskill]_] that combining classical shadows with classical
machine learning enables using ML methods to efficiently predict
properties of the quantum systems such as the expectation value of
Hamiltonian, correlations functions, entanglement entropies, etc.

.. figure::  /demonstrations/ml_classical_shadows/class_shadow_ml.png
   :align: center
   :alt: Combining ML with Classical Shadow

   Combining machine learning and classical shadow


In this demo, we combine classical shadow formalism with classical
machine learning and use them to predict the ground-state properties of
the 2-D antiferromagnetic Heisenberg model. So let's get started!

Building the 2-D Heisenberg Model
---------------------------------

In a two-dimensional antiferromagnetic Heisenberg model, each site of a
square lattice is occupied by a spin-1/2 particle. The Hamiltonian
associated with the model is:

.. math::  H = \sum_{\langle ij\rangle} J_{ij}(X_i X_j + Y_i Y_j + Z_i Z_j) 

Here, each term :math:`J_{ij}` refers to the coupling between spins
:math:`\sigma^{z}_{i}` and :math:`\sigma^{z}_{j}` and is sampled
uniformly from the interval [0, 2].

"""

import pennylane as qml
import pennylane.numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.notebook import tqdm


######################################################################
# To completely describe the model, we begin by defining the coupling
# matrix :math:`J` by providing the number of rows :math:`N_r` and columns
# :math:`N_c` present in the lattice. The total spins in the model will be
# :math:`N_r \times N_c`.
#


def build_coupling_mat(Nr, Nc):
    """Build the coupling matrix for the 2-D spin lattice of Heisenberg Model"""
    num_spins = Nr * Nc
    coupling_mat = np.zeros((num_spins, num_spins))
    for ind in range(num_spins):
        for idx in range(num_spins):
            if not coupling_mat[ind][idx]:
                if ((ind + 1) % Nc and idx - ind == 1) or (ind - idx == Nc):
                    coupling_mat[ind][idx] = 2 * np.random.rand()
                    coupling_mat[idx][ind] = coupling_mat[ind][idx]
    return coupling_mat


######################################################################
# Here we study the model with :math:`4` spins arranged on the nodes of a
# square lattice and require four qubits for the simulation. For this, we
# build the coupling matrix using ``build_coupling_mat``.
#

Nr, Nc = 2, 2
num_qubits = Nr * Nc
coupling_mat = build_coupling_mat(Nr, Nc)


######################################################################
# We can visualize the model as ``networkx`` graph by building it using
# the corresponding coupling matrix ``coupling_mat``.
#

G = nx.from_numpy_matrix(np.matrix(coupling_mat), create_using=nx.DiGraph)
pos = {i: (i % Nc, -(i // Nc)) for i in G.nodes()}
edge_labels = {(x, y): np.round(coupling_mat[x, y], 2) for x, y in G.edges()}
weights = [x + 1.5 for x in list(nx.get_edge_attributes(G, "weight").values())]

nx.draw(
    G,
    pos,
    node_color="lightblue",
    with_labels=True,
    node_size=600,
    width=weights,
    edge_color="firebrick",
)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
plt.show()


######################################################################
# We use the same coupling matrix :math:`J` to obtain the Hamiltonian
# :math:`H` for our sample model visualized above.
#


def H(x):
    """Build the Hamiltonian for the Heisenberg model"""
    coeffs, ops = [], []
    for ind in range(x.shape[0]):
        for idx in range(ind + 1, x.shape[0]):
            for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeff = x[ind, idx]
                if coeff:
                    coeffs.append(coeff)
                    ops.append(op(ind) @ op(idx))
    H = qml.Hamiltonian(coeffs, ops)
    return H


######################################################################
# For a Heisenberg model, a propetry of particular interest is usally the
# two-body correlation matrix :math:`C`, whose each element :math:`C_{ij}`
# is defined as the following expectation value of the following operator
# for each pair of spin :math:`\sigma^{z}_{i}` and :math:`\sigma^{z}_{j}`
# with respect to the ground state :math:`|\psi_{0}\rangle` of the model:
#
# .. math::  \hat{C}_{ij} = \frac{1}{3} (X_i X_j + Y_iY_j + Z_iZ_j)
#


def corr_function_op(ind, idx):
    """Build correlation function operator :math:`C_{ij}` for Heisenberg Model"""
    ops = []
    for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
        ops.append(op(ind) @ op(idx)) if ind != idx else ops.append(qml.Identity(idx))
    return ops


corr_function_op(1, 2)


######################################################################
# To calculate the exact ground state :math:`|\psi_{0}\rangle` of the
# model, we diagonalize the Hamiltonian :math:`H` and obtain the
# eigenvector corresponding to the smallest eigenvalue. We then build a
# circuit that will initialize the qubits into the quantum state given by
# this eigenvector and measure the expectation value for the provided set
# of observables.
#

ham = H(coupling_mat)
psi0 = np.zeros(2 ** (num_qubits))

if len(ham.ops):
    eigvals, eigvecs = sp.sparse.linalg.eigs(qml.utils.sparse_hamiltonian(ham))
    psi0 = eigvecs[:, np.argmin(eigvals)]

dev_exact = qml.device("default.qubit", wires=num_qubits)  # for exact simulation
dev_oshot = qml.device("default.qubit", wires=num_qubits, shots=1)  # for single-shot simulation


def circuit(psi, **kwargs):
    """circuit for measuring expectation value of an observables O_i with respect to state |psi>"""
    observables = kwargs.pop("observable")
    qml.QubitStateVector(psi / np.linalg.norm(psi), wires=range(int(np.log2(len(psi)))))
    return [qml.expval(o) for o in observables]


circuit_exact = qml.QNode(circuit, dev_exact)
circuit_oshot = qml.QNode(circuit, dev_oshot)


######################################################################
# Finally, to build the exact correlation matrix :math:`C`, we build each
# correlation operator :math:`\hat{C}_{ij}` and calculate its expectation
# value w.r.t the ground state :math:`|\psi_0\rangle`.
#

coups = list(it.product(range(num_qubits), repeat=2))
corrs = [corr_function_op(i, j) for i, j in coups]
expval_exact = np.zeros((num_qubits, num_qubits))

for i, j in coups:
    corrs = corr_function_op(i, j)
    if i == j:
        expval_exact[i][j] = 1.0
    else:
        expval_exact[i][j] = (
            np.sum(np.array([circuit_exact(psi0, observable=[o]) for o in corrs]).T) / 3
        )
        expval_exact[j][i] = expval_exact[i][j]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
shw = ax.imshow(expval_exact, cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1)
ax.xaxis.set_ticks(range(num_qubits))
ax.yaxis.set_ticks(range(num_qubits))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

cbar_ax = fig.add_axes([0.96, 0.124, 0.05, 0.754])
bar = fig.colorbar(shw, cax=cbar_ax)
bar.set_label(r"$C_{ij}$", fontsize=18, rotation=0)
bar.ax.tick_params(labelsize=14)
plt.show()


######################################################################
# Constructing Classical Shadows
# ------------------------------
#


######################################################################
# To build an an approximate classical representation of :math:`n`-qubit
# quantum state :math:`\rho`, we perform randomized single-qubit
# measurements on :math:`T`-copies of :math:`\rho` in Pauli basis
# :math:`X`, :math:`Y`, or :math:`Z` to yeild random :math:`n` pure
# product states :math:`|s_i\rangle` for each copy:
#
# .. math::  S_T(\rho) = \big\{|s_{i}^{(t)}\rangle: i\in\{1,\ldots, n\} t\in\{1,\ldots, T\} \big\} \in \{|0\rangle, |1\rangle, |+\rangle, |-\rangle, |i+\rangle, |i-\rangle\}
#
# Each of the :math:`|s_i^{(t)}\rangle` provides us with classical access
# to a single snapshot of the :math:`\rho`, and the :math:`nT`
# measurements yield the complete snapshot :math:`S_{T}` which requires
# just :math:`3nT` bits to be stored in the classical memory.
#


######################################################################
# .. figure::  /demonstrations/ml_classical_shadows/class_shadow_prep.png
#    :align: center
#    :alt: Preparing Classical Shadows
#
#    Workflow for classical shadow preparation
#


######################################################################
# To prepare classical shadow for the ground state of the Heisenberg
# model, we use ``circuit_oshot`` that prepares the ground state that we
# find using exact diagonalization of the Hamiltonian. To do so, it uses
# the ``qml.QubitStateVector`` operation for initializing the circuit in
# the provided state.
#

dev = qml.device("default.qubit", wires=num_qubits, shots=1)


@qml.qnode(dev)
def circuit(psi, **kwargs):
    qml.QubitStateVector(psi / np.linalg.norm(psi), wires=range(int(np.log2(len(psi)))))
    observables = kwargs.pop("observable")
    return [qml.expval(o) for o in observables]


######################################################################
# Now, we build the function ``gen_class_shadow`` that would make
# :math:`T` the classical shadows for the quantum state prepared by a
# given :math:`n`-qubit circuit.
#


def gen_class_shadow(circ_template, circuit_params, num_shadows, num_qubits):
    """
    Build classical shadow state for the given N-qubit system using T copies of
    randomized Pauli basis measurements.
    """
    # prepare the complete set of available Pauli operators
    unitary_ops = [qml.PauliX, qml.PauliY, qml.PauliZ]
    # sample random Pauli measurements uniformly
    unitary_ensmb = np.random.randint(0, 3, size=(num_shadows, num_qubits), dtype=int)

    meas_outcomes = np.zeros((num_shadows, num_qubits))
    for ns in range(num_shadows):
        # for each snapshot, extract the Pauli basis measurement to be performed
        meas_obs = [unitary_ops[unitary_ensmb[ns, i]](i) for i in range(num_qubits)]
        # perform single shot randomized Pauli measuremnt for each qubit
        meas_outcomes[ns, :] = circ_template(circuit_params, observable=meas_obs)

    return meas_outcomes, unitary_ensmb


shadow = gen_class_shadow(circuit_oshot, psi0, 100, num_qubits)
shadow[0][:5], shadow[1][:5]


######################################################################
# Furthermore, :math:`S_{T}` can be used to reconstruct the underlying
# n-qubit state :math:`\rho`:
#
# .. math::  \sigma_T(\rho) = \frac{1}{T} \sum_{1}^{T} \big(3|s_{1}^{(t)}\rangle\langle s_1^{(t)}| - \mathbb{I}\big)\otimes \ldots \otimes \big(3|s_{n}^{(t)}\rangle\langle s_n^{(t)}| - \mathbb{I}\big)
#


def snapshot_state(meas_list, obs_list):
    """Build the \sigma_T snapshot for reconstructing the quantum state from its classical shadow"""
    # undo the rotations done for performing Pauli measurements in the specific basis
    rotations = [
        qml.Hadamard(wires=0).matrix,  # X-basis
        qml.Hadamard(wires=0).matrix @ qml.S(wires=0).inv().matrix,  # Y-basis
        qml.Identity(wires=0).matrix,
    ]  # Z-basis

    # reconstruct snapshot from local Pauli measurements
    rho_snapshot = [1]
    for meas_out, basis in zip(meas_list, obs_list):
        # preparing state |s_i><s_i| using the post measurement outcome: |0><0| for 1 and |1><1| for -1
        state = np.array([[1, 0], [0, 0]]) if meas_out == 1 else np.array([[0, 0], [0, 1]])
        local_rho = 3 * (rotations[basis].conj().T @ state @ rotations[basis]) - np.eye(2)
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


def shadow_state_reconst(shadow):
    """Reconstruct the quantum state from its classical shadow by averaging over computed \sigma_T"""
    num_snapshots, num_qubits = shadow[0].shape
    meas_lists, obs_lists = shadow

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(meas_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots


######################################################################
# Let’s see how well does this reconstruction work for different value of
# :math:`T` by looking at the Fidelity :math:`\mathcal{F}` of the actual
# quantum state w.r.t the reconstructed quantum state from the classical
# shadow with :math:`T` copies. We see that on average, as the number of
# copies :math:`T` is increased, the reconstruction becomes more
# effective, and in the limit :math:`T\rightarrow\infty`, it will be
# exact.
#
# .. figure:: /demonstrations/ml_classical_shadows/fidel_snapshot.png
#    :align: center
#    :alt: Fidelity of reconstructed ground state with different shadow sizes :math:`T`
#
#    Fidelity of reconstructed ground state with different shadow sizes :math:`T`
#


######################################################################
# The reconstructed quantum state :math:`\sigma_T` can then be used to
# evaluate expectation values for some localized observable
# :math:`O = \bigotimes_{i}^{n} P_i`, where :math:`P_i \in \{I, X, Y, Z\}`
# using :math:`\text{Tr}(O\sigma_T)`. However, as shown above,
# :math:`\sigma_T` would be only an approximation of actual :math:`\rho`
# for finite values of :math:`T`. Therefore to estimate
# :math:`\langle O \rangle` robustly, we use the median of means
# estimation. For this purpose, we split up the :math:`T` shadows into
# some :math:`K` equally sized chunks and estimate the median of the mean
# value of :math:`\langle O \rangle` for each of these chunks.
#


def estimate_shadow_obervable(shadow, observable, k=10):
    """Estimate observable related to the quantum system using its classical shadow"""
    shadow_size, num_qubits = shadow[0].shape

    # convert Pennylane observables to indices
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_obs = np.array([map_name_to_int[observable.name]])
        target_locs = np.array([observable.wires[0]])
    else:
        target_obs = np.array([map_name_to_int[o.name] for o in observable.obs])
        target_locs = np.array([o.wires[0] for o in observable.obs])

    # perform median of means to return the result
    means = []
    meas_list, obs_lists = shadow
    for i in range(0, shadow_size, shadow_size // k):
        meas_list_k, obs_lists_k = (
            meas_list[i : i + shadow_size // k],
            obs_lists[i : i + shadow_size // k],
        )
        indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)
        if sum(indices):
            means.append(
                np.sum(np.prod(meas_list_k[indices][:, target_locs], axis=1)) / sum(indices)
            )
        else:
            means.append(0)

    return np.median(means)


######################################################################
# Let us try to estimate the correlation matrix :math:`C` from the
# classical shadow of our Heisenberg model this time.
#

coups = list(it.product(range(num_qubits), repeat=2))
corrs = [corr_function_op(i, j) for i, j in coups]
qbobs = [x for sublist in corrs for x in sublist]
expval_estmt = np.zeros((num_qubits, num_qubits))

shadow = gen_class_shadow(circuit_oshot, psi0, 1000, num_qubits)

failure_rate = 1.0
k = int(2 * np.log(2 * len(qbobs) / failure_rate))

for i, j in coups:
    corrs = corr_function_op(i, j)
    if i == j:
        expval_estmt[i][j] = 1.0
    else:
        expval_estmt[i][j] = (
            np.sum(np.array([estimate_shadow_obervable(shadow, o, k=k + 1) for o in corrs])) / 3
        )
        expval_estmt[j][i] = expval_estmt[i][j]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
shw = ax.imshow(expval_estmt, cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1)
ax.xaxis.set_ticks(range(num_qubits))
ax.yaxis.set_ticks(range(num_qubits))
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

cbar_ax = fig.add_axes([0.96, 0.124, 0.05, 0.754])
bar = fig.colorbar(shw, cax=cbar_ax)
bar.set_label(r"$C_{ij}$", fontsize=18, rotation=0)
bar.ax.tick_params(labelsize=14)
plt.show()


######################################################################
# Training Classical Machine Learning Models
# ------------------------------------------
#


######################################################################
# There are multiple ways in which one can combine classical shadows and
# classical machine learning. This could include training various machine
# learning models to learn the classical representation of quantum systems
# based on some system parameter, estimating a property from such learned
# classical representations, or a combination of both. In our case, we
# consider the problem of using infinite-width neural networks to learn
# the ground-state representation of Heisenberg model :math:`H(x_l)` from
# the coupling vector :math:`x_l` and predict the correlation functions
# :math:`C_{ij}`:
#
# .. math::  \big\{x_l \rightarrow \sigma_T(\rho(x_l)) \rightarrow \text{Tr}(\hat{C}_{ij} \sigma_T(\rho(x_l))) \big\}_{l=1}^{N}
#
# Using the theory of infinite-width neural networks [[#neurtangkernel]_], we
# consider the following form of classical machine learning models:
#
# .. math::  \hat{\sigma}_{N} (x) = \sum_{l=1}^{N} \kappa(x, x_l)\sigma_T (x_l) = \sum_{l=1}^N \left(\sum_{l^{\prime}=1}^{N} k(x, x_{l^{\prime}})(K+\lambda I)^{-1}_{l, l^{\prime}} \sigma_T(x_l) \right),
#
# where :math:`\lambda > 0` is a regularization paramter in cases when
# :math:`K` is not invertible, :math:`\sigma_T(x_l)` denotes the classical
# representation of the ground state :math:`\rho(x_l)` of the Heisenberg
# model constructed using :math:`T` randomized Pauli measurements, and
# :math:`K_{ij}=k(x_i, x_j)` is the kernel matrix with
# :math:`k(x, x^{\prime})` as the kernel function.
#
# Similarly, to estimate a property on the predicted ground state
# :math:`\sigma_T(x_l)` using the trained ML model can then be done by
# evaluating:
#
# .. math::  \text{Tr}(\hat{O} \hat{\sigma}_{N} (x)) = \sum_{l=1}^{N} \kappa(x, x_l)\text{Tr}(O\sigma_T (x_l)),
#
# Here, we train the classical kernel-based ML models using :math:`N = 70`
# randomly chosen value of coupling matrices :math:`J` with
# :math:`J_{ij} \in [0, 2]` for predicting the correlation functions
# :math:`C_{ij}`.
#

# Neural tangent kernel
import jax
from neural_tangents import stax

# Traditional ML methods and techniques
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


######################################################################
# First, to build the dataset, we use the function ``build_dataset`` that
# takes input as the size of the dataset (``num_points``), the topology of
# the lattice (:math:`Nr` and :math:`Nc`), and the number of randomized
# Pauli measurements (T) for construction of classical shadows. The
# ``X_data`` is the set of coupling vectors which are defined as a
# stripped version of coupling matrix :math:`J`, where only non-duplicate
# and non-zero :math:`J_{ij}` are considered. The ``y_exact`` and
# ``y_clean`` are the set of correlation vectors, i.e., the flattened
# correlation matrix :math:`C`, computed w.r.t ground-state obtained from
# exact diagonalization and classical shadow representation (with
# :math:`T=1000`), respectively.
#


def build_dataset(num_points, Nr, Nc, T=500):
    """Builds dataset for Heisenberg model: X (copling vector), y (correlation matrix)"""

    num_qubits = Nr * Nc
    X, y_exact, y_estim = [], [], []

    for idx in tqdm(range(num_points), leave=False):

        coupling_mat = build_coupling_mat(Nr, Nc)
        ham = H(coupling_mat)

        psi = np.zeros(2 ** num_qubits)
        if len(ham.ops):  # Sanity Check
            eigvals, eigvecs = sp.sparse.linalg.eigs(qml.utils.sparse_hamiltonian(ham))
            psi = eigvecs[:, np.argmin(eigvals)]

        shadow = gen_class_shadow(circuit_oshot, psi, T, num_qubits)

        coups = list(it.product(range(num_qubits), repeat=2))
        corrs = [corr_function_op(i, j) for i, j in coups]
        qbobs = [x for sublist in corrs for x in sublist]

        failure_rate = 1
        k = int(2 * np.log(2 * len(qbobs) / failure_rate))
        expval_exact = np.zeros((num_qubits, num_qubits))
        expval_estim = np.zeros((num_qubits, num_qubits))
        for i, j in coups:
            corrs = corr_function_op(i, j)
            if i == j:
                expval_exact[i][j], expval_estim[i][j] = 1.0, 1.0
            else:
                expval_exact[i][j] = (
                    np.sum(np.array([circuit_exact(psi, observable=[o]) for o in corrs]).T) / 3
                )
                expval_estim[i][j] = (
                    np.sum(np.array([estimate_shadow_obervable(shadow, o, k=k + 1) for o in corrs]))
                    / 3
                )
                expval_exact[j][i], expval_estim[j][i] = expval_exact[i][j], expval_estim[i][j]

        coupling_vec = []
        for coup in coupling_mat.reshape(1, -1)[0]:
            if coup and coup not in coupling_vec:
                coupling_vec.append(coup)

        X.append(np.array(coupling_vec))
        y_exact.append(expval_exact.reshape(1, -1)[0])
        y_estim.append(expval_estim.reshape(1, -1)[0])

    return np.array(X), np.array(y_exact), np.array(y_estim)


X, y_exact, y_estim = build_dataset(100, Nr, Nc, 1000)

X_data, y_data = X, y_exact
X_data.shape, y_data.shape, y_exact.shape


######################################################################
# Now that we have our dataset ready. We shift our focus to the ML models.
# Here, we use a set of three different Kernel functions: (i) Gaussian
# Kernel, (ii) Dirichlet Kernel, and (iii) Neural Tangent Kernel. For all
# three of them, we consider the regularization parameter :math:`\lambda`
# from the following set:
#
# .. math::  \lambda = \left\{ 0.0025, 0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.0, 5.0, 10.0 \right\}
#
# Next, we define the kernel functions :math:`k(x, x^{\prime})` for each
# of the mentioned kernel:
#


######################################################################
# .. math::  k(x, x^{\prime}) = e^{-\gamma|| x - x^{\prime}||^{2}_{2}} \tag{Gaussian Kernel}
#
# For Gaussian kernel, the hyperaparemeter
# :math:`\gamma = N^{2}/\sum_{i=1}^{N}\sum_{j=1}^{N}|| x_i-x_j||^2_2 > 0`
# is chosen to be the inverse of the average distance :math:`x_i` and
# :math:`x_j` and the kernel is implemented using the Radial-basis
# function (rbf) kernel in the ``sklearn`` library.
#


######################################################################
# .. math::  k(x, x^{\prime}) = \sum_{i\neq j}\sum_{k_i=-5}^{5}\sum_{k_j=-5}^{5} \cos{\big(\pi(k_i(x_i-x_i^{\prime}) + k_j(x_j-x_j^{\prime}))\big)} \tag{Dirichlet Kernel}
#
# Dirichlet kernel is motivated by writing the :math:`\text{k}^{th}`
# partial sum of the Fourier series of function :math:`f` as a
# convolution. Here, we define this kernel as ``kernel_dirichlet`` for
# :math:`k=11` as follows.
#

## Dirichlet kernel ##
kernel_dirichlet = np.zeros((X_data.shape[0], 11 * X_data.shape[1]))
for idx in range(len(X_data)):
    for k in range(len((X_data[idx]))):
        for k1 in range(-5, 6):
            kernel_dirichlet[idx, 11 * k + k1 + 5] += np.cos(np.pi * k1 * X_data[idx][k])
kernel_dirichlet.shape


######################################################################
# .. math::  k(x, x^{\prime}) = k^{\text{NTK}}(x, x^{\prime}) \tag{Neural Tangent Kernel}
#
# The neural tangent kernel :math:`k^{\text{NTK}}` used here is equivalent
# to an infinite-width feed-forward neural network with four hidden
# layers, and that uses the rectified linear unit (ReLU) as the activation
# function. This is implemented using the ``neural_tangents`` library.
#

## Neural tangent kernel ##
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(1),
)
kernel_NN = kernel_fn(X_data, X_data, "ntk")

for i in range(len(kernel_NN)):
    for j in range(len(kernel_NN)):
        kernel_NN[i][j] /= (kernel_NN[i][i] * kernel_NN[j][j]) ** 0.5

kernel_NN.shape


######################################################################
# From the three defined kernel methods, to obtain the best ML model, we
# perform hyperparameter tuning using cross-validation for the prediction
# task of each :math:`C_{ij}`. For this purpose, we implement the function
# ``fit_predict_data``, which takes input as the correlation function
# index ``cij``, kernel matrix ``kernel`` and internal kernel mapping
# ``opt`` required by Epsilon-Support Vector and Kernel-Ridge Regressions
# functions from ``sklearn`` library.
#


def fit_predict_data(cij, kernel, opt="linear"):
    """Trains the dataset via hyperparameter tuning and returns the prediction from the best model"""

    # perform instance-wise normalization to get k(x, x')
    for idx in range(len(kernel)):
        kernel[idx] /= np.linalg.norm(kernel[idx])

    # training data (estimated from measurement data)
    y = np.array([y_estim[i][cij] for i in range(len(X_data))])
    X_train, X_test, y_train, y_test = train_test_split(kernel, y, test_size=0.3, random_state=24)

    # testing data (exact expectation values)
    y_clean = np.array([y_exact[i][cij] for i in range(len(X_data))])
    _, _, _, y_test_clean = train_test_split(kernel, y_clean, test_size=0.3, random_state=24)

    # hyperparameter tuning with cross validation
    models = [
        (lambda Cx: svm.SVR(kernel=opt, C=Cx, epsilon=0.1)),  # Epsilon-Support Vector Regression
        (lambda Cx: KernelRidge(kernel=opt, alpha=1 / (2 * Cx))),
    ]  # Kernel-Ridge based Regression
    hyperparams = [
        0.0025,
        0.0125,
        0.025,
        0.05,
        0.125,
        0.25,
        0.5,
        1.0,
        5.0,
        10.0,
    ]  # Regularization parameter
    best_model, best_pred, best_cv_score, best_test_score = None, None, np.inf, np.inf
    for model in models:
        for hyperparam in hyperparams:
            cv_score = -np.mean(
                cross_val_score(
                    model(hyperparam), X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
                )
            )
            if best_cv_score > cv_score:
                best_model = model(hyperparam).fit(X_train, y_train)
                best_pred = best_model.predict(X_test)
                best_cv_score = cv_score
                best_test_score = np.linalg.norm(
                    best_model.predict(X_test).ravel() - y_test_clean.ravel()
                ) / (len(y_test) ** 0.5)

    return (
        best_model,
        best_pred,
        y_test_clean,
        np.round(best_cv_score, 5),
        np.round(best_test_score, 5),
    )


######################################################################
# We perform the fitting and prediction for each :math:`C_{ij}` and print
# the output in a tabular format.
#

kernel_list = ["Gaussian kernel", "Dirichlet kernel", "Neural Tangent kernel"]
kernel_data = np.zeros((num_qubits ** 2, len(kernel_list), 2))
y_predclean, y_predicts1, y_predicts2, y_predicts3 = [], [], [], []

for cij in tqdm(range(num_qubits ** 2), leave=False):
    clf, y_predict, y_clean, best_cv_score, test_score = fit_predict_data(cij, X_data, opt="rbf")
    y_predclean.append(y_clean)
    kernel_data[cij][0] = (best_cv_score, test_score)
    y_predicts1.append(y_predict)
    clf, y_predict, y_clean, best_cv_score, test_score = fit_predict_data(cij, kernel_dirichlet)
    kernel_data[cij][1] = (best_cv_score, test_score)
    y_predicts2.append(y_predict)
    clf, y_predict, y_clean, best_cv_score, test_score = fit_predict_data(cij, kernel_NN)
    kernel_data[cij][2] = (best_cv_score, test_score)
    y_predicts3.append(y_predict)

# For each C_ij print (best_cv_score, test_score) pair
row_format = "{:>10}{:>22}{:>23}{:>25}"
print(row_format.format("", *kernel_list))
for idx, data in enumerate(kernel_data):
    print(
        row_format.format(
            f"C_{idx//num_qubits}{idx%num_qubits} \t| ", str(data[0]), str(data[1]), str(data[2])
        )
    )


######################################################################
# Overall, we find that the model with Gaussian kernel performed the best,
# while the Dirichlet kernel ones performed the worst for predicting the
# expectation value of the correlation function :math:`C_{ij}` for the
# ground state of the Heisenberg model. However, the best choice of
# :math:`\lambda` differed substantially across the different
# :math:`C_{ij}` for all the kernels. This means that no particular choice
# of the hyperparameter :math:`\lambda` could perform better than others
# at an average. We present the predicted correlation matrix
# :math:`C^{\prime}` for randomly selected Heisenberg models from the test
# set below for comparison against the actual correlation matrix
# :math:`C`, which is obtained from the ground state found using exact
# diagonalization.
#

fig, axes = plt.subplots(3, 4, figsize=(22, 12))
corr_vals = [y_predclean, y_predicts1, y_predicts2, y_predicts3]
plt_plots = [2, 1, 23]

cols = [
    "From {}".format(col)
    for col in ["Exact Diagnalization", "RBF Kernel", "Dirichlet Kernel", "Neural Tangent Kernel"]
]
rows = ["Model {}".format(row) for row in plt_plots]

for ax, col in zip(axes[0], cols):
    ax.set_title(col, fontsize=18)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, rotation=90, fontsize=24)

for itr in range(3):
    for idx, corr_val in enumerate(corr_vals):
        shw = axes[itr][idx].imshow(
            np.array(corr_vals[idx]).T[plt_plots[itr]].reshape(Nr * Nc, Nr * Nc),
            cmap=plt.get_cmap("RdBu"),
            vmin=-1,
            vmax=1,
        )
        axes[itr][idx].xaxis.set_ticks(range(Nr * Nc))
        axes[itr][idx].yaxis.set_ticks(range(Nr * Nc))
        axes[itr][idx].xaxis.set_tick_params(labelsize=18)
        axes[itr][idx].yaxis.set_tick_params(labelsize=18)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
bar = fig.colorbar(shw, cax=cbar_ax)
bar.set_label(r"$C_{ij}$", fontsize=18, rotation=0)
bar.ax.tick_params(labelsize=16)
plt.show()


######################################################################
# Finally, we also try to showcase the effect of a bigger training data
# size :math:`N` and a larger number of randomized Pauli measurements
# :math:`T`. For this, we look at the average value of ``best_cv_score``
# for each model, which gives the RMSE (root-mean-square error) error for
# prediction of :math:`C_ij`. The first plot looks at the different
# training sizes :math:`N` with a fixed number of randomized Pauli
# measurements :math:`T=100`. In comparison, the second plot looks at
# different shadow sizes :math:`T` with a fixed training data size
# :math:`N = 70`. In both cases, the performance improvement saturates
# after a sufficient increase in :math:`N` and :math:`T` values for all
# three kernels.
#


######################################################################
# .. figure::  /demonstrations/ml_classical_shadows/rmse_shadaow.png
#     :width: 48 %
#
# .. figure::  /demonstrations/ml_classical_shadows/rmse_training.png
#     :width: 48 %
#
# Predicting two-point correlation functions for ground state of
# 2D antiferromagnetic Heisenberg model over different training size :math:`N`
# and different shadow size :math:`T`.


######################################################################
# .. _ml_classical_shadow_references:
#
# References
# ----------
#
# .. [#preskill]
#
#    H. Y. Huang, R. Kueng, G. Torlai, V. V. Albert, J. Preskill, "Provably
#    efficient machine learning for quantum many-body problems",
#    `arXiv:2106.12627 [quant-ph] (2021)
#     <https://arxiv.org/abs/2106.12627>`__
#
# .. [#neurtangkernel]
#
#    A. Jacot, F. Gabriel, and C. Hongler. "Neural tangent kernel:
#    Convergence and generalization in neural networks". `NeurIPS, 8571–8580
#    (2018) <https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf>`__
#
