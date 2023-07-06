
import jax
import optax
from jax import numpy as jnp
import pennylane as qml
import numpy as np
jax.config.update("jax_enable_x64", True)

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


X = np.loadtxt("embeddings_metric_learning/X_antbees.txt")  # inputs
Y = np.loadtxt("embeddings_metric_learning/Y_antbees.txt")  # labels
X_val = np.loadtxt(
    "embeddings_metric_learning/X_antbees_test.txt"
)  # validation inputs
Y_val = np.loadtxt("embeddings_metric_learning/Y_antbees_test.txt")  # validation labels
Y[Y == 0] = -1  # rename label 0 to -1
Y_val[Y_val == 0] = -1

scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)
X_val = scaler.fit_transform(X_val)

# split data into two classes
A = X[Y == -1]
B = X[Y == 1]
A_val = X_val[Y_val == -1]
B_val = X_val[Y_val == 1]

n_features = 2
n_qubits = 2 * n_features + 1

dev = qml.device("lightning.qubit", wires=n_qubits)

@jax.jit
@qml.qnode(qml.device("default.qubit", wires=2))
def overlap(q_weights, x1, x2):
    qml.QAOAEmbedding(features=x1, weights=q_weights, wires=[0, 1])
    qml.adjoint(qml.QAOAEmbedding)(features=x2, weights=q_weights, wires=[0, 1])

    return qml.expval(qml.Projector(np.array([0, 0]), wires=[0, 1]))


def overlaps(params, X1=None, X2=None):
    res = 0
    for x1 in X1:
        for x2 in X2:
            x1_feats = params["params_classical"] @ x1
            x2_feats = params["params_classical"] @ x2
            res += overlap(params["params_quantum"], x1_feats, x2_feats)
    return res / (len(X1) * len(X2))


def cost_fn(params, A=None, B=None):
    aa = overlaps(params, X1=A, X2=A)
    bb = overlaps(params, X1=B, X2=B)
    ab = overlaps(params, X1=A, X2=B)

    d_hs = -ab + 0.5 * (aa + bb)
    return 1 - d_hs


# generate initial parameters for circuit (4 layers)
init_pars_quantum = jnp.array(np.random.normal(loc=0, scale=0.1, size=(4, 3)))

# generate initial parameters for linear layer
init_pars_classical = jnp.array(np.random.normal(loc=0, scale=0.1, size=(2, 512)))

params = {"params_quantum": init_pars_quantum, "params_classical": init_pars_classical}


def get_batch(batch_size, A, B):
    selectA = np.random.choice(range(len(A)), size=(batch_size,), replace=False)
    selectB = np.random.choice(range(len(B)), size=(batch_size,), replace=False)
    return A[selectA], B[selectB]


@jax.jit
def update(params, opt_state, A, B):
    loss, grads = jax.value_and_grad(cost_fn)(params, A, B)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)
batch_size = 4

A_monitor, B_monitor = get_batch(20, A, B)
A_monitor_val, B_monitor_val = get_batch(20, A_val, B_val)

for i in range(100000):
    A_batch, B_batch = get_batch(batch_size, A, B)
    params, opt_state, loss = update(params, opt_state, A_batch, B_batch)

    if i % 1000 == 0:
        # monitor progress on the 20 validation samples
        test_loss = cost_fn(params, A_monitor, B_monitor)
        train_loss = cost_fn(params, A_monitor_val, B_monitor_val)

        print(f"Step {i} Estimated loss train: {train_loss} | test: {test_loss}")

A_B_train = np.r_[A_monitor, B_monitor]

gram = [[overlaps(params, X1=[x1], X2=[x2]) for x1 in A_B_train] for x2 in A_B_train]
ax = plt.subplot(111)
im = ax.matshow(gram, vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


A_B_val = np.r_[A_monitor_val, B_monitor_val]

gram = [[overlaps(params, X1=[x1], X2=[x2]) for x1 in A_B_val] for x2 in A_B_val]
ax = plt.subplot(111)
im = ax.matshow(gram, vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


A_2d = (params["params_classical"] @ A.T).T
plt.scatter(A_2d[:, 0], A_2d[:, 1], c="red", s=5, label="Ants - training")

B_2d = (params["params_classical"] @ B.T).T
plt.scatter(B_2d[:, 0], B_2d[:, 1], c="blue", s=5, label="Bees - training")

A_val_2d = (params["params_classical"] @ A_val.T).T
plt.scatter(A_val_2d[:, 0], A_val_2d[:, 1], c="orange", s=10, label="Ants - validation")

B_val_2d = (params["params_classical"] @ B_val.T).T
plt.scatter(B_val_2d[:, 0], B_val_2d[:, 1], c="green", s=10, label="Bees - validation")

plt.legend()
plt.show()


x_new = A_val[0]
print(x_new.shape)


A_examples, B_examples = get_batch(20, A, B)

# compute the distance between class examples and new input
o_A = overlaps(params, X1=A_examples, X2=[x_new])
o_B = overlaps(params, X1=B_examples, X2=[x_new])

# weigh the mean distances by the class label
prediction = -1 * jnp.mean(o_A) + 1 * jnp.mean(o_B)

print(prediction)
