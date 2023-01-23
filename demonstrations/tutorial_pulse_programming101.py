r"""Simulating differentialble pulse programs in PennyLane
==========================================================

.. meta::
    :property="og:description": Simulating differentialble pulse programs in PennyLane
    :property="og:image": https://pennylane.ai/qml/_images/pauli_shadows.jpg

*Author: Korbinian Kottmann — Posted: 20 February 2023.

In this demo we are going to introduce pulse gates and differentiable pulse programming, showcase
the current functionality in PennyLane and run the ctrl-VQE algorithm for an example molecule.

Topic1
------
Reference docs :doc:`tutorial_classical_shadows`
Reference literature, ctrl vqe paper MET [#Asthana2022]_
.. math:: \sum_i c_i k_i

Pulse programming
-----------------
Introduce pulses, channels, pulse gates, time dependent Schrodinger equation

Pulse gates in PennyLane
------------------------
Implementation of the above concepts in PennyLane

Gradients of pulse gates
------------------------
Can compute them on hardware with parameter shift rule [#Leng2022]_


"""

import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt


##############################################################################
# More text


import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax

import optax # for jax optimization

import matplotlib.pyplot as plt
from datetime import datetime

symbols = ["H", "H"]
coordinates = np.array([[0., 0., 0.],[0., 0., 1.5]])

basis_set = "sto-3g"
H, n_wires = qml.qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    charge=0,
    mult=1,
    basis=basis_set,
    mapping="bravyi_kitaev",
    method="pyscf",
)

print(f"no qubits: {n_wires}")
coeffs, obs = jnp.array(H.coeffs), H.ops
H_obj = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
H_obj_m = qml.matrix(H_obj)
E_exact = np.min(np.linalg.eigvalsh(H_obj_m))

##############################################################################
# We are considering a coupled transmon qubit system with the constant drift term Hamiltonian 
#
# .. math:: H_D = \sum_q \omega_q a_q^\dagger a_q + \sum_q \frac{\delta_q}{2} a^\dagger_q a^\dagger_q a_q a_q + \sum_{\braket{pq}} g_{pq} a^\dagger_p a_q
# 
# with bosonic creation and annihilation operators. We are only going to consider qubit subspace such that the quadratic term is zero.
# The order of magnitude of the resonance frequencies $\omega_q$ and coupling strength $g_{pq}$ are taken from [#Asthana2022]_.

def a(wires):
    return 0.5*qml.PauliX(wires) + 0.5j* qml.PauliY(wires)
def ad(wires):
    return 0.5*qml.PauliX(wires) - 0.5j* qml.PauliY(wires)

omega = 4.8 * jnp.ones(n_wires)
g = 0.01 * jnp.ones(n_wires-1)

H_D = qml.op_sum(*[qml.s_prod(omega[i], ad(i) @ a(i)) for i in range(n_wires)])
H_D += qml.op_sum(*[qml.s_prod(g[i], ad(i) @ a(i+1)) for i in range(n_wires-1)])

##############################################################################
# The system is driven under the control term
#
# .. math:: H_C(t) = \sum_q \Omega_q(t) \left(e^{i\nu_q t} a_q + e^{-i\nu_q t} a^\dagger_q \right)
# 
# with the (real) time-dependent amplitude :math:`\Omega(t)` and frequency :math:`\nu_q` of the drive.

# TODO use official convenience functions once merged
def pwc(t1, t2):
    def wrapped(params, t):
        N = len(params)
        idx = jnp.array(N/(t2 - t1) * (t - t1), dtype=int) # corresponding sample
        return params[idx]

    return wrapped

def f(t1, t2, sgn=1.):
    # assuming p = (len(t_bins) + 1) for the frequency nu
    def wrapped(p, t):
        return pwc(t1, t2)(p[:-1], t) * jnp.exp(sgn*1j*p[-1]*t)
    return wrapped

t1 = 0.
t2 = 15.

fs = [f(t1, t2, 1.) for i in range(n_wires)]
fs += [f(t1, t2, -1.) for i in range(n_wires)]
ops = [a(i) for i in range(n_wires)]
ops += [ad(i) for i in range(n_wires)]
# ops = [qml.PauliX(i) for i in range(n_wires)]
# ops += [qml.PauliY(i) for i in range(n_wires)]

H_C = qml.ops.dot(fs, ops)

H_pulse = H_D + H_C
t_bins = 200 # number of time bins
p = jnp.array([jnp.ones(t_bins + 1) for _ in range(n_wires)])
dev = qml.device("default.qubit", wires=range(n_wires))

@jax.jit
@qml.qnode(dev, interface="jax")
def qnode(p, t=15.):
    qml.evolve(H_pulse)(params=(*p, *p), t=t)
    return qml.expval(H_obj)


def abs_diff(p):
    """compute |p_i - p_i-1|^2"""
    #p = jnp.array(p)
    return jnp.mean(jnp.abs(jnp.diff(p, axis=1))**2)
def cost_fn(p):
    C_exp = qnode(p)
    C_par = jnp.mean(jnp.abs(p)**2)
    C_der = abs_diff(p)
    return C_exp + C_par + C_der
theta = jnp.array([jnp.ones(t_bins) for _ in range(n_wires)])

n_epochs = 50
cosine_decay_scheduler = optax.cosine_decay_schedule(0.5, decay_steps=n_epochs, alpha=0.95)
optimizer = optax.adam(learning_rate=0.1) # cosine_decay_scheduler
opt_state = optimizer.init(theta)

value_and_grad = jax.jit(jax.value_and_grad(cost_fn, argnums=0))

energy = np.zeros(n_epochs)
cost = np.zeros(n_epochs)
theta_i = [theta]

## Optimization loop

t0 = datetime.now()
_ = value_and_grad(theta)
t1 = datetime.now()
print(f"grad and val compilation time: {t1 - t0}")

for n in range(n_epochs):
    val, grad_circuit = value_and_grad(theta)
    updates, opt_state = optimizer.update(grad_circuit, opt_state)
    theta = optax.apply_updates(theta, updates)

    energy[n] = qnode(theta)
    cost[n] = val
    theta_i.append(theta)

    if not n%5:
        print(f"{n+1} / {n_epochs}; energy: {energy[n]}; cost: {val}")

# TODO: how can the energy be smaller than the exact diagonalization one
fig, axs = plt.subplots(ncols=2, figsize=(10,4))
ax = axs[0]
ax.plot(cost,".:", label="cost")
ax.plot(energy, "x:", label="energy")
#ax.plot(cost - energy, "x:", label="control contribution")
ax.plot([0, n_epochs], [E_exact]*2, "--", label="exact", color="grey")
ax.legend()

ax = axs[1]
for i in range(n_wires):
    ax.plot(theta[i],"-", label=f"qubit {i}", alpha=0.5)
ax.legend()
plt.show()

##############################################################################
#
# Conclusion
# ----------
# Conclusion
#
#
# References
# ----------
#
# .. [#Asthana2022]
#
#     Ayush Asthana, Chenxu Liu, Oinam Romesh Meitei, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall
#     "Minimizing state preparation times in pulse-level variational molecular simulations."
#     `arXiv:2203.06818 <https://arxiv.org/abs/2203.06818>`__, 2022.
#
# .. [#Leng2022] 
# 
#     Jiaqi Leng, Yuxiang Peng, Yi-Ling Qiao, Ming Lin, Xiaodi Wu
#     Differentiable Analog Quantum Computing for Optimization and Control
#     `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`__, 2022#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
