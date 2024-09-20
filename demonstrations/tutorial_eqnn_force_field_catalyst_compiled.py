r"""Compiling "Symmetry-invariant quantum machine learning force fields" with PennyLane-Catalyst
========================================================


To speed up our training process, we can use PennyLane-Catalyst to compile our training workflow.

As opposed to jax.jit, catalyst.qjit innately understands PennyLane quantum instructions and performs better on
Lightning backend.


"""

import pennylane as qml
import numpy as np

import jax
from jax import numpy as jnp

import matplotlib.pyplot as plt

import catalyst
from catalyst import qjit

from jax.example_libraries import optimizers
from sklearn.preprocessing import MinMaxScaler

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1.0j], [1.0j, 0]])
Z = np.array([[1, 0], [0, -1]])

sigmas = jnp.array(np.array([X, Y, Z]))  # Vector of Pauli matrices
sigmas_sigmas = jnp.array(
    np.array(
        [np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)]  # Vector of tensor products of Pauli matrices
    )
)

def singlet(wires):
    # Encode a 2-qubit rotation-invariant initial state, i.e., the singlet state.
    qml.Hadamard(wires=wires[0])
    qml.PauliZ(wires=wires[0])
    qml.PauliX(wires=wires[1])
    qml.CNOT(wires=wires)


def equivariant_encoding(alpha, data, wires):
    # data (jax array): cartesian coordinates of atom i
    # alpha (jax array): trainable scaling parameter
    hamiltonian = jnp.einsum("i,ijk", data, sigmas)  # Heisenberg Hamiltonian
    U = jax.scipy.linalg.expm(-1.0j * alpha * hamiltonian / 2)
    qml.QubitUnitary(U, wires=wires, id="E")


def trainable_layer(weight, wires):
    hamiltonian = jnp.einsum("ijk->jk", sigmas_sigmas)
    U = jax.scipy.linalg.expm(-1.0j * weight * hamiltonian)
    qml.QubitUnitary(U, wires=wires, id="U")


# Invariant observbale
Heisenberg = [
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.PauliZ(0) @ qml.PauliZ(1),
]
Observable = qml.Hamiltonian(np.ones((3)), Heisenberg)


def noise_layer(epsilon, wires):
    for _, w in enumerate(wires):
        qml.RZ(epsilon[_], wires=[w])


D = 6  # Depth of the model
B = 1  # Number of repetitions inside a trainable layer
rep = 2  # Number of repeated vertical encoding

active_atoms = 2  # Number of active atoms
                  # Here we only have two active atoms since we fixed the oxygen (which becomes non-active) at the origin
num_qubits = active_atoms * rep


# We need to use "lightning.qubit" device for Catalyst compilation.
dev = qml.device("lightning.qubit", wires=num_qubits)


######################################################################
# The core function that is called repeatedly can benefit from being just-in-time compiled with qjit.
# All we need to do is decorate the function with the `@qjit` decorator.
#
# Catalyst has its own `for_loop` function to work with qjit.
# `catalyst.for_loop` should be used when the loop bounds or step depend on the qjit-ted function's input arguments.
# If there is no such dependence, `catalyst.for_loop` can still be used.
# Here we showcase both usages.

@qjit
@qml.qnode(dev)
def vqlm_qjit(data, params):
    weights = params["params"]["weights"]
    alphas = params["params"]["alphas"]
    epsilon = params["params"]["epsilon"]
    # Initial state
    @catalyst.for_loop(0, rep, 1)
    def singlet_loop(i):
        singlet(wires=jnp.arange(active_atoms)+active_atoms*i)
    singlet_loop()
    # Initial encoding
    for i in range(num_qubits):
        equivariant_encoding(
            alphas[i, 0], jnp.asarray(data)[i % active_atoms, ...], wires=[i]
        )        
    # Reuploading model
    for d in range(D):
        qml.Barrier()
        for b in range(B):
            # Even layer
            for i in range(0, num_qubits - 1, 2):
                trainable_layer(weights[i, d + 1, b], wires=[i, (i + 1) % num_qubits])
            # Odd layer
            for i in range(1, num_qubits, 2):
                trainable_layer(weights[i, d + 1, b], wires=[i, (i + 1) % num_qubits])
        # Symmetry-breaking
        if epsilon is not None:
            noise_layer(epsilon[d, :], range(num_qubits))
        # Encoding
        for i in range(num_qubits):
            equivariant_encoding(
                alphas[i, d + 1], jnp.asarray(data)[i % active_atoms, ...], wires=[i]
            )
    return qml.expval(Observable)

# vectorizing for batched training with `catalyst.vmap`
vec_vqlm = catalyst.vmap(vqlm_qjit, in_axes=(0, {'params': {'alphas': None, 'epsilon': None, 'weights': None}} ), out_axes=0)

# loss function for cost
def mse_loss(predictions, targets):
    return jnp.mean(0.5 * (predictions - targets) ** 2)

# Compile a training step
# many calls so compile = faster!
@qjit
def train_step(step_i, opt_state, loss_data):

    def cost(weights, loss_data):
        data, E_target, F_target = loss_data
        E_pred = vec_vqlm(data, weights)
        l = mse_loss(E_pred, E_target)
        return l

    net_params = get_params(opt_state)
    loss = cost(net_params, loss_data)
    grads = catalyst.grad(cost, method = "fd", h=1e-13, argnums=0)(net_params, loss_data)
    return loss, opt_update(step_i, grads, opt_state)


# Return prediction and loss at inference times, e.g. for testing
@qjit
def inference(loss_data, opt_state):
    data, E_target, F_target = loss_data
    net_params = get_params(opt_state)
    E_pred = vec_vqlm(data, net_params)
    l = mse_loss(E_pred, E_target)
    return E_pred, l


#################### main ##########################
### setup ###
# Load the data
energy = np.load("eqnn_force_field_data/Energy.npy")
forces = np.load("eqnn_force_field_data/Forces.npy")
positions = np.load(
    "eqnn_force_field_data/Positions.npy"
)  # Cartesian coordinates shape = (nbr_sample, nbr_atoms,3)
shape = np.shape(positions)


### Scaling the energy to fit in [-1,1]

scaler = MinMaxScaler((-1, 1))

energy = scaler.fit_transform(energy)
forces = forces * scaler.scale_


# Placing the oxygen at the origin
data = np.zeros((shape[0], 2, 3))
data[:, 0, :] = positions[:, 1, :] - positions[:, 0, :]
data[:, 1, :] = positions[:, 2, :] - positions[:, 0, :]
positions = data.copy()

forces = forces[:, 1:, :]  # Select only the forces on the hydrogen atoms since the oxygen is fixed


# Splitting in train-test set
indices_train = np.random.choice(np.arange(shape[0]), size=int(0.8 * shape[0]), replace=False)
indices_test = np.setdiff1d(np.arange(shape[0]), indices_train)

E_train, E_test = (energy[indices_train, 0], energy[indices_test, 0])
F_train, F_test = forces[indices_train, ...], forces[indices_test, ...]
data_train, data_test = (
    jnp.array(positions[indices_train, ...]),
    jnp.array(positions[indices_test, ...]),
)

### training ###
opt_init, opt_update, get_params = optimizers.adam(1e-2)

np.random.seed(42)
weights = np.zeros((num_qubits, D, B))
weights[0] = np.random.uniform(0, np.pi, 1)
weights = jnp.array(weights)

# Encoding weights
alphas = jnp.array(np.ones((num_qubits, D + 1)))

# Symmetry-breaking (SB)
np.random.seed(42)
epsilon = jnp.array(np.random.normal(0, 0.001, size=(D, num_qubits)))
epsilon = None  # We disable SB for this specific example
epsilon = jax.lax.stop_gradient(epsilon)  # comment if we wish to train the SB weights as well.



net_params = {"params": {"weights": weights, "alphas": alphas, "epsilon": epsilon}}
opt_state = opt_init(net_params)
running_loss = []


num_batches = 5000 # number of optimization steps
batch_size =  256  # number of training data per batch

batch = np.random.choice(np.arange(np.shape(data_train)[0]), batch_size, replace=False)
loss_data = data_train[batch, ...], E_train[batch, ...], F_train[batch, ...]


# The main training loop
# We call `train_step` and `inference` many times, so the speedup from qjit will be quite significant!
for ibatch in range(num_batches):
    # select a batch of training points
    batch = np.random.choice(np.arange(np.shape(data_train)[0]), batch_size, replace=False)

    # preparing the data
    loss_data = data_train[batch, ...], E_train[batch, ...], F_train[batch, ...]
    loss_data_test = data_test, E_test, F_test

    # perform one training step
    loss, opt_state = train_step(num_batches, opt_state, loss_data)

    # computing the test loss and energy predictions
    E_pred, test_loss = inference(loss_data_test, opt_state)
    running_loss.append([float(loss), float(test_loss)])


history_loss = np.array(running_loss)

### plotting ###
fontsize = 12
plt.figure(figsize=(4,4))
plt.plot(history_loss[:, 0], "r-", label="training error")
plt.plot(history_loss[:, 1], "b-", label="testing error")

plt.yscale("log")
plt.xlabel("Optimization Steps", fontsize=fontsize)
plt.ylabel("Mean Squared Error", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.show()


plt.figure(figsize=(4,4))
plt.title("Energy predictions", fontsize=fontsize)
plt.plot(energy[indices_test], E_pred, "ro", label="Test predictions")
plt.plot(energy[indices_test], energy[indices_test], "k.-", lw=1, label="Exact")
plt.xlabel("Exact energy", fontsize=fontsize)
plt.ylabel("Predicted energy", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.show()
