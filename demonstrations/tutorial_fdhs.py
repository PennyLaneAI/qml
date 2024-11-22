r"""Fixed Depth Hamiltonian Simulation via Cartan Decomposition
===============================================================

abstract

Introduction
------------

The KAK theorem is an important result from Lie theory that states that any Lie group element :math:`U` can be decomposed
as :math:`U = K_1 A K_2`, where :math:`K_{1, 2}` and :math:`A` are elements of two special sub-groups
:math:`\mathcal{K}` and :math:`\mathcal{A}`, respectively. You can think of this KAK decomposition as a generalization of
the singular value decomposition to Lie groups.

For that, recall that the singular value decomposition states that any
matrix :math:`M \in \mathbb{C}^{m \times n}` can be decomposed as :math:`M = U \Lambda V^\dagger`, where :math:`\Lambda`
are the diagonal singular values and :math:`U \in \mathbb{C}^{m \times \mu}` and :math:`V^\dagger \in \mathbb{C}^{\mu \times n}`
are left- and right-unitary with :math:`\mu = \min(m, n)`.

In the case of the KAK decomposition, :math:`\mathcal{A}` is an Abelian subgroup such that all its elements are commuting,
just as is the case for diagonal matrices.

We can use this general result from Lie theory as a powerful circuit decomposition technique.

Goal
----

Unitary gates in quantum computing are described by the special orthogonal Lie group :math:`SU(2^n)`, so we can use the KAK
theorem to decompose quantum gates into :math:`U = K_1 A K_2`. While the mathematical statement is rather straight-forward,
actually finding this decomposition is not. We are going to follow the recipe prescribed in 
`Fixed Depth Hamiltonian Simulation via Cartan Decomposition <https://arxiv.org/abs/2104.00728>`__ [#Kökcü]_, 
that tackles this decomposition on the level of the associated Lie algebra via Cartan decomposition.

In particular, we are going to consider the problem of time-evolving a Hermitian operator :math:`H` the generates the time-evolution unitary :math:`U = e^{-i t H}`.
We are going to perform a special case of KAK decomposition, a "KhK decomposition" if you will, on the algebra level of H in terms of

.. math:: H = K^\dagger h_0 K.

This then induces the KAK decomposition on the group level as

.. math:: e^{-i t H} = K^\dagger e^{-i t h_0} K.

Let us walk through an explicit example, doing theory and code side-by-side.
For that we are going to use the Heisenberg model generators and Hamiltonian for :math:`n=4` qubits.
The foundation to a KAK decomposition is a Cartan decomposition of the associated Lie algebra :math:`\mathfrak{g}`.
For that, let us first construct it and import some libraries that we are going to use later.


"""
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pennylane as qml
from pennylane import X, Y, Z

import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

n_wires = 4
gens = [X(i) @ X(i+1) for i in range(n_wires-1)]
gens += [Y(i) @ Y(i+1) for i in range(n_wires-1)]
gens += [Z(i) @ Z(i+1) for i in range(n_wires-1)]

H = qml.sum(*gens)

g = qml.lie_closure(gens)
g = [op.pauli_rep for op in g]

##############################################################################
# 
# Cartan decomposition
# --------------------
# 
# A Cartan decomposition is a bipartition :math:`\mathfrak{g} = \mathcal{k} \oplus \mathcal{m}` into a vertical subspace
# :math:`\mathfrak{k}` and an orthogonal horizontal subspace :math:`\mathfrak{m}`. In practice, it can be induced by an
# involution function :math:`\Theta` that fulfils :math:`\Theta(\Theta(g)) = g \forall g \in \mathfrak{g}`. Different 
# involutions lead to different types of Cartan decompositions, which have been fully classified by Cartan 
# (see `wikipedia <https://en.wikipedia.org/wiki/Symmetric_space#Classification_result>`__).
# 
# .. note::
#     Note that :math:`\mathfrak{k}` is the small letter k in
#     `Fraktur <https://en.wikipedia.org/wiki/Fraktur>`__ and a 
#     common - not our - choice for the vertical subspace in a Cartan decomposition.
#
# One common choice of involution is the so-called even-odd involution for Pauli words
# :math:`P = P_1 \otimes P_2 .. \otimes P_n` where :math:`P_j \in \{I, X, Y, Z\}`.
# It essentially counts whether the number of non-identity Pauli operators in the Pauli word is even or odd.

def even_odd_involution(op):
    """Generalization of EvenOdd to sums of Paulis"""
    [pw] = op.pauli_rep
    parity = len(pw) % 2

    return parity

even_odd_involution(X(0)), even_odd_involution(X(0) @ Y(3))

##############################################################################
# 
# The vertical and horizontal subspaces are the two eigenspaces of the involution, corresponding to the :math:`\pm 1` eigenvalues.
# In particular, we have :math:`\Theta(\mathfrak{k}) = \mathfrak{k}` and :math:`\Theta(\mathfrak{m}) = - \mathfrak{m}`.
# So in order to perform the Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`, we simply
# sort the operators by whether or not they yield a plus or minus sign from the involution function.

def cartan_decomposition(g, involution):
    """Cartan Decomposition g = k + m
    
    Args:
        g (List[PauliSentence]): the (dynamical) Lie algebra to decompose
        involution (callable): Involution function :math:`\Theta(\cdot)` to act on PauliSentence ops, should return ``0/1`` or ``True/False``.
    
    Returns:
        k (List[PauliSentence]): the even parity subspace :math:`\Theta(\mathfrak{k}) = \mathfrak{k}`
        m (List[PauliSentence]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}` """
    m = []
    k = []

    for op in g:
        if involution(op): # odd parity
            k.append(op)
        else: # even parity
            m.append(op)
    return k, m

k, m = cartan_decomposition(g, even_odd_involution)
len(g), len(k), len(m)


##############################################################################
# We have successfully decomposed the :math:`60`-dimensional Lie algebra 
# into a :math:`24`-dimensional vertical subspace and a :math:`36`-dimensional subspace.
#
# Note that not every bipartition of a Lie algebra constitutes a Cartan decomposition.
# For that, the subspaces need to fulfil the following three commutation relations
#
# .. math::
#     \begin{align}
#     [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} & \text{ (subalgebra)}\\
#     [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} & \text{ (reductive property)}\\
#     [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k} & \text{ (symmetric property)}
#     \end{align}
#
# In particular, :math:`\mathfrak{k}` is closed under commutation and is therefore a subalgebra, whereas :math:`\mathfrak{m}` is not.
# This also has the consequence that the associated Lie group :math:`\mathcal{K} := e^{i \mathfrak{k}}` is a subgroup
# of the associated Lie group :math:`\mathcal{G} := e^{i \mathfrak{g}}`.
#
# Cartan subalgebra
# -----------------
# 
# With this we have identified the first subgroup (:math:`\mathcal{K}`) of the KAK decomposition. The other subgroup
# is induced by the so-called (horizontal) Cartan subalgebra :math:`\mathfrak{h}`. This is a maximal Abelian subalgebra of :math:`\mathfrak{m}` and is not unique.
# For the case of Pauli words we can simply pick any element in :math:`\mathfrak{m}` and collect all other operators in :math:`\mathfrak{m}`
# that commute with it.
#
# We then obtain a further split of the vector space :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h}`,
# where :math:`\tilde{\mathfrak{m}}` is just the remainder of :math:`\mathfrak{m}`.

def _commutes_with_all(candidate, ops):
    """Check if ``candidate commutes with all ``ops``"""
    for op in ops:
        com = candidate.commutator(op)
        com.simplify()
        
        if not len(com) == 0:
            return False
    return True

def cartan_subalgebra(m, which=0):
    """Compute the Cartan subalgebra from the odd parity space :math:`\mathfrak{m}` of the Cartan decomposition

    This implementation is specific for cases of bases of m with pure Pauli words as detailed in Appendix C in `2104.00728 <https://arxiv.org/abs/2104.00728>`__.
    
    Args:
        m (List[PauliSentence]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}
        which (int): Choice for initial element of m from which to construct the maximal Abelian subalgebra
    
    Returns:
        mtilde (List): remaining elements of :math:`\mathfrak{m}` s.t. :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h}`.
        a (List): Cartan subalgebra

    """

    h = [m[which]] # first candidate
    mtilde = m.copy()

    for m_i in m:
        if _commutes_with_all(m_i, h):
            if m_i not in h:
                h.append(m_i)
    
    for h_i in h:
        mtilde.remove(h_i)
    
    return mtilde, h

mtilde, h = cartan_subalgebra(m)
len(g), len(k), len(mtilde), len(h)

##############################################################################
# We now have the Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \tilde{\mathfrak{m}} \oplus \mathfrak{h}``
# and with that all the necessary ingredients for the KAK decomposition.
# 
# Variational KhK
# ---------------
#
# Obtaining the actual decomposition is highly non-trivial and there is no canonical way to go about computing it in terms of linear algebra sub-routines.
# In [#Kökcü]_, the authors propose to find the extrema of the cost function
# 
# .. math:: f(\theta) = \langle K(\theta) v K(\theta)^\dagger, H\rangle
# 
# where :math:`\langle \cdot, \cdot \rangle` is some inner product (in our case the trace inner product :math:`\langle A, B \rangle = \text{tr}(A^\dagger B)`).
# This construction uses the operator :math:`v = \sum_j \pi^j h_j \in \mathfrak{h}`
# that is such that :math:`e^{i t v}` is dense in :math:`e^{i \mathcal{h}}`.
# The latter means that for any point in :math:`e^{i \mathcal{h}}` there is a :math:`t` such that :math:`e^{i t v}` reaches it.
# Let us construct it.

gammas = [np.pi**i for i in range(len(h))]

v = qml.dot(gammas, h)
v_m = qml.matrix(v, wire_order=range(n_wires))
v_m = jnp.array(v_m)

##############################################################################
# 
# This procedure has the advantage that we can use an already decomposed ansatz
# 
# .. math:: K(\theta) = \prod_j e^{-i \theta_j k_j}
# 
# for the vertical unitary.
# 
# Now we just have to define the cost function and find an extremum.
# In this case we are going to use gradient descent to minimize the cost function to a minimum.
# We are going to use ``jax`` and ``optax`` and write some boilerplate for the optimization procedure.

def run_opt(
    value_and_grad,
    theta,
    n_epochs=500,
    lr=0.1,
):
    """Boilerplate jax optimization"""
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(theta)

    energy = []
    gradients = []
    thetas = []

    @jax.jit
    def step(opt_state, theta):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta, val

    t0 = datetime.now()
    ## Optimization loop
    for _ in range(n_epochs):
        opt_state, theta, val = step(opt_state, theta)

        energy.append(val)
        thetas.append(theta)
        
    t1 = datetime.now()
    print(f"final loss: {val}; min loss: {np.min(energy)}; after {t1 - t0}")

    return thetas, energy, gradients


##############################################################################
# We can now implement the cost function and find a minimum via gradient descent.

H_m = qml.matrix(H, wire_order=range(n_wires))
H_m = jnp.array(H_m)

def K(theta, k):
    for th, k_j in zip(theta, k):
        qml.exp(-1j * th * k_j.operation())

@jax.jit
@jax.value_and_grad
def loss(theta):
    K_m = qml.matrix(K, wire_order=range(n_wires))(theta, k)
    A = K_m @ v_m @ K_m.conj().T
    return jnp.trace(A.conj().T @ H_m).real

theta0 = jnp.ones(len(k), dtype=float)

thetas, energy, _ = run_opt(loss, theta0, n_epochs=500)
plt.plot(energy)
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()

##############################################################################
# This gives us the optimal values of the parameters :math:`\theta_\text{opt}` of :math:`K(\theta_\text{opt}) =: K_c`.

theta_opt = thetas[-1]
Kc_m = qml.matrix(K, wire_order=range(n_wires))(theta_opt, k)

##############################################################################
# The special element :math:`h_0` from the Cartan subalgebra :math:`\mathfrak{h}` is given by
# rotating the Hamiltonian by the critical :math:`K_c`.
# 
# .. math:: h_0 = K_c H K_c^\dagger.

h_0_m = Kc_m @ H_m @ Kc_m.conj().T
h_0 = qml.pauli_decompose(h_0_m)
print(len(h_0))

# assure that h_0 is in \mathfrak{h}
h_vspace = qml.pauli.PauliVSpace(h)
not h_vspace.is_independent(h_0.pauli_rep, tol=1e-2)

##############################################################################
#
# This (trivially) gives us the KhK decomposition of :math:`H`,
# 
# .. math:: H = K_c^\dagger h_0 K_c
# 
# This trivially reproduces the original Hamiltonian.
#

H_re = Kc_m.conj().T @ h_0_m @ Kc_m
np.allclose(H_re, H_m)

##############################################################################
# We can now check if the Hamiltonian evolution is reproduced correctly.
#

t = 1.
U_exact = qml.exp(-1j * t * H)
U_exact_m = qml.matrix(U_exact, wire_order=range(n_wires))

def U_kak(theta_opt, t):
    K(theta_opt, k)
    qml.exp(-1j * t * h_0)
    qml.adjoint(K)(theta_opt, k)

U_kak_m = qml.matrix(U_kak, wire_order=range(n_wires))(theta_opt, t)

def trace_distance(A, B):
    return 1 - np.abs(np.trace(A.conj().T @ B))/len(A)

trace_distance(U_exact_m, U_kak_m)






##############################################################################
# Indeed we find that the KAK decomposition that we found reproduces the unitary evolution operator.
# Note that this is valid for arbitrary :math:`t`, such that in that sense the Hamiltonian simulation operator has a fixed depth.

##############################################################################
# Time evolutions
# ---------------
# 
# We can compute multiple time evolutions and see that the 
#

ts = jnp.linspace(0.2, 1., 10)

Us_exact = jax.vmap(lambda t: qml.matrix(qml.exp(-1j * t * H), wire_order=range(n_wires)))(ts)

def Us_kak(t):
    return Kc_m.conj().T @ jax.scipy.linalg.expm(-1j * t * h_0_m) @ Kc_m

Us_kak = jax.vmap(Us_kak)(ts)
Us_trotter50 = jax.vmap(lambda t: qml.matrix(qml.TrotterProduct(H, time=t, n=50, order=4), wire_order=range(n_wires)))(ts)
Us_trotter500 = jax.vmap(lambda t: qml.matrix(qml.TrotterProduct(H, time=t, n=500, order=4), wire_order=range(n_wires)))(ts)

res_kak = 1 - jnp.abs(jnp.einsum("bij,bji->b", Us_exact.conj(), Us_kak)) / 2**n_wires
res_trotter50 = 1 - jnp.abs(jnp.einsum("bij,bji->b", Us_exact.conj(), Us_trotter50)) / 2**n_wires
res_trotter500 = 1 - jnp.abs(jnp.einsum("bij,bji->b", Us_exact.conj(), Us_trotter500)) / 2**n_wires

plt.plot(ts, res_kak, label="KAK")
plt.plot(ts, res_trotter50, "x--", label="50 Trotter steps")
plt.plot(ts, res_trotter500, ".-", label="500 Trotter steps")
plt.ylabel("empirical error")
plt.xlabel("t")
# plt.yscale("log")
plt.legend()
plt.show()


##############################################################################
# The KAK decomposition is particularly well-suited for smaller systems as the circuit depth is equal to the
# dimension of the subspaces, in particular :math:`2 |\mathfrak{k}| + |\mathfrak{h}|`. Note, however,
# that these dimensions typically scale exponentially in the system size.
#


##############################################################################
# 
# Conclusion
# ----------
#
# We learned about the powerful and versatile tool of KAK circuit decomposition and applied it to
# time evolution operators.
#



##############################################################################
# 
# References
# ----------
#
# .. [#Kökcü]
#
#     Efekan Kökcü, Thomas Steckmann, Yan Wang, J. K. Freericks, Eugene F. Dumitrescu, Alexander F. Kemper
#     "Fixed Depth Hamiltonian Simulation via Cartan Decomposition"
#     `arXiv:2104.00728 <https://arxiv.org/abs/2104.00728>`__, 2021.
#
# .. [#Wiersma]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Meyer]
#
#     Johannes Jakob Meyer, Marian Mularski, Elies Gil-Fuster, Antonio Anna Mele, Francesco Arzani, Alissa Wilms, Jens Eisert
#     "Exploiting symmetry in variational quantum machine learning"
#     `arXiv:2205.06217 <https://arxiv.org/abs/2205.06217>`__, 2022.
#
# .. [#Nguyen]
#
#     Quynh T. Nguyen, Louis Schatzki, Paolo Braccia, Michael Ragone, Patrick J. Coles, Frederic Sauvage, Martin Larocca, M. Cerezo
#     "Theory for Equivariant Quantum Neural Networks"
#     `arXiv:2210.08566 <https://arxiv.org/abs/2210.08566>`__, 2022.
#
# .. [#Fontana]
#
#     Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, Marco Pistoia
#     "The Adjoint Is All You Need: Characterizing Barren Plateaus in Quantum Ansätze"
#     `arXiv:2309.07902 <https://arxiv.org/abs/2309.07902>`__, 2023.
#
# .. [#Ragone]
#
#     Michael Ragone, Bojko N. Bakalov, Frédéric Sauvage, Alexander F. Kemper, Carlos Ortiz Marrero, Martin Larocca, M. Cerezo
#     "A Unified Theory of Barren Plateaus for Deep Parametrized Quantum Circuits"
#     `arXiv:2309.09342 <https://arxiv.org/abs/2309.09342>`__, 2023.
#
# .. [#Goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
# .. [#Somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
