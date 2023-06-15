r"""Evidence for the utility of quantum computing before fault tolerance
========================================================================

.. meta::
    :property="og:description": Evidence for the utility of quantum computing before fault tolerance
    :property="og:image": https://pennylane.ai/qml/_images/diffable_mitigation_thumb.png

.. related::

    tutorial_error_mitigation Error mitigation with Mitiq and PennyLane
    tutorial_diffable-mitigation Differentiable quantum error mitigation

*Author: Korbinian Kottmann — Posted: 16  June 2023.*

abstract / attention grabber

Introduction
============

We know that quantum computers can do things that classical computers can not. Most prominently, 
there is Shor's algorithm for prime factoring. This algorithm necessitates fault tolerance via 
error correction, which is not yet feasible with currently available machines. One highly debated 
question in the field is whether or not noisy devices we have access to _right now_ are already 
useful or can outperform a classical computer for certain tasks. For the latter point, 
demonstrations of quantum computational advantage have been put forward with the 2019 Sycamore 
(Google), the 2020 Jiuzhang (Chinese academy of science) and the 2022 Borealis (Xanadu) experiments. 
These demonstrations, however, are only of limited practical utility. A new quest has been set out 
for _practical_ quantum computational advantage. That is, an application at which a quantum computer 
accelerates compared to the best known classical methods _and_ that has a useful application. In the 
new article [#ibm] by a team of scientists at IBM, a case is made that with their latest device 
comprising 127 qubits and record-breaking coherence times, they can faithfully simulate the time 
dynamics of a complex quantum many-body system. One of the key achievements of the paper is the 
successful application of error mitigation on a large system and demonstrating that it can yield 
faithful results even in very noisy scenarios with modestly deep circuits. In this demo we are going
to explain the error mitigation method used in the paper and discuss its implications.

Problem setting
===============
Before we go into the details of the error mitigation methods, let us breifly summarize the problem 
setting. The authors of [#ibm] are concerned with simulating the time dynamics of the 2D transverse field Ising model  

.. math:: H = -J \sum_{\langle qp \rangle}Z_q Z_p + h \sum_q X_q

with nearest neighbor interactions matching the topology of the device. 
The time evolution is approximated by trotterization of the time evolution operator

.. math:: U(T) \approx \left(\prod_{\langle qp \rangle} e^{i \delta t J Z_q Z_p} \prod_{q} e^{-i \delta t J X_q} \right)^{\frac{T}{\delta t}}

for evolution time $T$ and trotter step size $\delta t$. That means the circuit of concern here is a 
series of consecutive $\text{RZZ}(\theta_J)$ and $\text{RX}(\theta_h)$ rotations. The corresponding 
angles are related to the physical parameters via $\theta_J = -2J \delta t$ and $\theta_h = 2h \delta t$. 
From here on we are going to focus just on the values of $\theta_h$ and keep $\theta_J=-\pi/2$ fixed 
(in the paper is due to the simpliciation this introduces in the decomposition of the $\text{RZZ}$ gate).

The complexity of the classical simulation varies with the parameter $\theta_h$. For the extrema 
$\theta_h=0$ and $\theta_h=\pi/2$, the system becomes trivially solvable. We interpolate between 
those extrema and show the final value of a single weight observable $\langle Z_4\rangle$.
"""
import pennylane as qml
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

n_wires = 9
# Describe noise
noise_gate = qml.DepolarizingChannel
noise_strength = 0.01

# Load devices
dev_ideal = qml.device("default.mixed", wires=n_wires)
dev_noisy = qml.transforms.insert(noise_gate, noise_strength, position="all")(dev_ideal)

# 3x3 grid with nearest neighbors
connections = [(0, 1), (1, 2),
               (3, 4), (4, 5),
               (6, 7), (7, 8),
               (0, 3), (3, 6),
               (1, 4), (4, 7),
               (2, 5), (5, 8)]
connections = [(i, i+1) for i in range(n_wires-1)]

def time_evolution(theta_h, n_layers = 10, obs = qml.PauliZ(4)):
    for _ in range(n_layers):
        for i, j in connections:
            qml.IsingZZ(-jnp.pi/2, wires=(i, j))
        for i in range(n_wires):
            qml.RX(theta_h, wires=i)
    return qml.expval(obs)

qnode_ideal = qml.QNode(time_evolution, dev_ideal, interface="jax")
qnode_noisy = qml.QNode(time_evolution, dev_noisy, interface="jax")

thetas = jnp.linspace(0, jnp.pi/2, 20)

res_ideal = jax.vmap(qnode_ideal)(thetas)
res_noisy = jax.vmap(qnode_noisy)(thetas)

plt.plot(thetas, res_ideal, label="exact")
plt.plot(thetas, res_noisy, label="noisy")
plt.xticks([0, jnp.pi/8, jnp.pi/4, 3*jnp.pi/8, jnp.pi/2], ["0", "$\\pi$/8", "$\\pi/4$", "$3\\pi/4$", "$\\pi/2$"])
plt.xlabel("$\\theta_h$")
plt.ylabel("$\\langle Z_4 \\rangle$")
plt.legend()
plt.show()



##############################################################################
# Error mitigation via zero noise extrapolation
# =============================================
# 
# Error mitigation is the process of retrieving more accurate information via classical post-processing
# of noisy quantum executions. The authors in [#ibm] employ zero noise extrapolation (ZNE), which serves as
# a biased estimator of expectation values. The idea of ZNE is fairly straight forward: Imagine we want to
# obtain the exact quantum function :math:`f` that estimaes an expectation value under noiseless evolution.
# However, we only have access to a noisy version :math:`f^{⚡}`. Now imagine we can controllably increase 
# the noise present in terms of some parameter :math:`G`, the noise gain, where :math:`G=1` corresponds to
# the default noise present in the device. In ZNE, we evaluate :math:`f^{⚡}` at increasing values of :math:`G`
# from which we can extrapolate back to zero noise :math:`G=0` via a suitable curve fit.

from pennylane.transforms import mitigate_with_zne

scale_factors = [1, 2, 3]

qnode_mitigated = mitigate_with_zne(
    scale_factors=scale_factors,
    folding=qml.transforms.fold_global,
    extrapolate=qml.transforms.richardson_extrapolate,
)(qnode_noisy)

res_mitigated = jax.vmap(qnode_mitigated)(thetas)
plt.plot(thetas, res_ideal, label="exact")
plt.plot(thetas, res_mitigated, label="mitigated")
plt.plot(thetas, res_noisy, label="noisy")
plt.xticks([0, jnp.pi/8, jnp.pi/4, 3*jnp.pi/8, jnp.pi/2], ["0", "$\\pi$/8", "$\\pi/4$", "$3\\pi/4$", "$\\pi/2$"])
plt.xlabel("$\\theta_h$")
plt.ylabel("$\\langle Z_4 \\rangle$")
plt.legend()
plt.show()


##############################################################################
# Comparison with classical methods
# =================================
# The authors of [#ibm] compare their experimental results with classical methods. For this, they consider three 
# scenarios of different classical complexity. The main ingredient is the parameter $\Theta_h$ which determines 
# the angle of the single qubit $\text{RX}(\Theta_h)$ rotations of the trotter layers, as well as the overall 
# circuit structure.
#
# For $\Theta_h=-\pi/2$ (case 1) the dynamics become trivial with just a global phase factor introduced, such that s
# tarting from the $|0\rangle^{\otimes 127}$ initial state, the expectation values $\langle Z_q \rangle$ are trivially 
# one at all times. This serves as an anchor point of orientation. Varying $\Theta_h$ (case 2) then increases the 
# classical simulation complexity. However, for the circuits chosen, it is still possible to simulate the dynamical 
# expectation values of local observables by taking into account their light-cone in the evolution with reduced depth 
# (note that these are not full state vector evolutions but rather just directly looking at the dynamical expectation 
# values of interest). In the third and most compelx case, the circuit is altered such that the light-cone trick from 
# before does not work anymore.
# 
# One of the points of the papers is to compare the experiments with sophisticated classical simulation methods. 
# The authors chose tensor networks, in particular matrix product states (MPS) and isometric tensor network states 
# (isoTNS) for simulation. MPS are native to one dimension topologies, but are still often employed for two 
# dimensional systems as is the case here. THe justification for that is their low computational complexity and 
# decades of algorithmic development and optimization. Better suited to the given problem are isoTNS, which are 
# restrictions of projected entangled pair states (PEPS) with some simplifications reducing the high computational 
# and algorithmic complexity, at the cost of more approximation errors.
# 
# In both cases, the so-called bond-dimension $\chi$, a hyper parameter chosen by the user, directly determines 
# the bipartite entanglement entropy these state can capture. It is known that due to the area law of entanglement, 
# many ground states of relevant physical system can be faithfully approximated with such tensor network states 
# with finite bond dimension. However, that is generally not the case for time dynamics as the entanglement entropy 
# grows linearly and the area law no longer holds. Therefore, the employed tensor network methods are doomed for 
# most dynamical simulations, as is the case the in the [#ibm].
# 
# `It can be argued <https://twitter.com/gppcarleo/status/1669251392156860418>`_ that there are better suited 
# classical algorithms for these kind of dynamical simulations, 
# `with neural quantum states being one of them <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.
# Further, the full flagship of 2D tensor network methods have not been employed here, 
# though achieving that is arguably beyond the scope of this experimental demonstration, though worth noting.
# 
#
# Conclusion
# ----------
# 
# conclusion
#
#
# References
# ----------
#
# .. [#ibm]
#
#     Youngseok Kim, Andrew Eddins, Sajant Anand, Ken Xuan Wei, Ewout van den Berg, Sami Rosenblatt, Hasan Nayfeh, Yantao Wu, Michael Zaletel, Kristan Temme & Abhinav Kandala 
#     "Evidence for the utility of quantum computing before fault tolerance"
#     `Nature 618, 500–505 <https://www.nature.com/articles/s41586-023-06096-3>`__, 2023.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt

