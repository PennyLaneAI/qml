r"""

.. _general_parshift:

General parameter-shift rules for quantum gradients
===================================================

.. meta::

    :property="og:description": Reconstruct quantum functions and compute their derivatives.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_genpar.png

.. related::

   tutorial_rotoselect Leveraging trigonometry to choose circuits with Rotoselect
   tutorial_quantum_analytic_descent Building multivariate models with QAD


*Author: David Wierichs (Xanadu resident). Posted: ?? August 2021.*

In this demo we will learn how to reconstruct univariate quantum functions, i.e., those that
depend on a single parameter. This reconstruction and variants thereof allow us to derive more
general parameter-shift rules to compute the derivatives of any order.
Furthermore, there are some optimization techniques that use such reconstructions.
All we will need for the demo is the insight that these functions are Fourier series in their
variable, and a bit of knowledge about Fourier transforms.
In the most general case, the shift rules become even simpler conceptually (but are a bit more
computation-intensive).

We will briefly recap the derivation of the functional form below, the full
version can be found together with details on the reconstruction idea, derivations of the
parameter-shift rules, and considerations for multivariate functions in the paper
`General parameter-shift rules for quantum gradients <https://arxiv.org/abs/2107.12390>`_
[#GenPar]_.
The core idea to consider these quantum functions as Fourier series was first presented in
the preprint
`Calculus on parameterized quantum circuits <https://arxiv.org/abs/1812.06323>`_ [#CalcPQC]_,
we will follow [#GenPar]_, but there also are two preprints discussing general parameter-shift
rules: an algebraic approach in
`Analytic gradients in variational quantum algorithms: Algebraic extensions of the parameter-shift rule to general unitary transformations <https://arxiv.org/abs/2107.08131>`_ [#AlgeShift]_
and one focusing on special gates and spectral decompositions, namely
`Generalized quantum circuit differentiation rules <https://arxiv.org/abs/2108.01218>`_
[#GenDiffRules]_.

|

.. figure:: ../demonstrations/general_parshift/thumbnail_genpar.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Function reconstruction and differentiation via parameter shifts.

.. note ::

    Before going through this tutorial, we recommend that readers refer to the
    :doc:`Rotosolve & Rotoselect tutorial </demos/tutorial_rotoselect>`, and having a basic
    understanding of the :doc:`parameter-shift rule </glossary/parameter_shift>` might make
    this tutorial easier to dive into.

Cost functions arising from quantum gates
-----------------------------------------
We will consider cost functions from parametrized quantum circuits with a single variational
parameter.
For this we will use a certain gate structure that allows us to tune the number of eigenvalues
of the parametrized unitary, and thus the number of frequencies in the cost function with respect
to the variational parameter.
More concretely, we initialize the qubit register in a random state :math:`|\psi\rangle`
(for which we will make sure the seed is fixed) and apply a layer of Pauli-:math:`Z` rotations
``RZ`` to all qubits, where all rotations are parametrized by the same angle :math:`x`.
We then measure the expectation value of a random Hermitian observable :math:`B`.

Let's start by defining functions that generate the random initial state :math:`|\psi\rangle`
and the random observable :math:`B` for a given number of qubits :math:`N` and a fixed seed:
"""


from scipy.stats import unitary_group
import numpy.random as rnd


def random_state(N, seed):
    """Create a random state on N qubits."""
    states = unitary_group.rvs(2 ** N, random_state=rnd.default_rng(seed))
    return states[0]


def random_observable(N, seed):
    """Create a random observable on N qubits."""
    rnd.seed(seed)
    # Generate real and imaginary part separately and (anti-)symmetrize them for Hermiticity
    real_part, imag_part = rnd.random((2, 2 ** N, 2 ** N))
    real_part += real_part.T
    imag_part -= imag_part.T
    return real_part + 1j * imag_part


###############################################################################
# Let's make sure this gives us a valid, normalized state of dimension :math:`2^N` and a Hermitian
# matrix with size :math:`2^N\times 2^N`.
# As we will use JAX later on, we use its NumPy implementation from the beginning, enabling 64-bit
# ``float`` precision via the JAX config.


from jax.config import config

config.update("jax_enable_x64", True)
import jax
from jax import numpy as np

# Number of qubits
N = 4
# Test random state
psi = random_state(N, 1234)
print("psi is normalized:       ", np.isclose(np.linalg.norm(psi), 1.0))
print("psi has shape (2**N,):   ", psi.shape == (2 ** N,))
# Test random observable
B = random_observable(N, 1234)
print("B is Hermitian:          ", np.allclose(B, B.T.conj()))
print("B has shape (2**N, 2**N):", B.shape == (2 ** N, 2 ** N))


###############################################################################
# Now let's set up a "circuit generator", namely a function that will create a ``device`` and the
# ``cost`` function using :math:`|\psi\rangle` as initial state and measuring :math:`B`, depending
# on the number of qubits:


import pennylane as qml


def make_cost(N, seed):
    """Create a cost function on N qubits with N frequencies."""
    dev = qml.device("default.qubit", wires=N)

    @jax.jit
    @qml.qnode(dev, interface="jax")
    def cost(x):
        """Cost function on N qubits with N frequencies."""
        qml.QubitStateVector(random_state(N, seed), wires=dev.wires)
        for w in dev.wires:
            qml.RZ(x, wires=w)
        return qml.expval(qml.Hermitian(random_observable(N, seed), wires=dev.wires))

    return cost


###############################################################################
# Let's also prepare some plotting functionalities and colors:


import matplotlib.pyplot as plt

# Set the plotting range on the x-axis
xlim = (-np.pi, np.pi)
X = np.linspace(*xlim, 60)
# Colors
green = "#209494"
orange = "#ED7D31"
red = "xkcd:brick red"
blue = "xkcd:cerulean"
pink = "xkcd:bright pink"


###############################################################################
# Now that we took care of these preparations, let's dive right into it:
# It can rather easily be shown [#GenPar]_ that the output of this circuit, namely the function
#
# .. math ::
#
#   E(x)=\langle\psi | U^\dagger(x) B U(x)|\psi\rangle
#
# with :math:`U(x)` summarizing the ``RZ`` gates,
#
# .. math ::
#
#   U(x)=\prod_{a=1}^N R_Z^{(a)}(x) = \prod_{a=1}^N \exp\left(-i\frac{x}{2} Z_a\right),
#
# takes the form of a Fourier series in the variable :math:`x`. That is to say that
#
# .. math ::
#
#   E(x) = a_0 + \sum_{\ell=1}^R a_{\ell}\cos(\ell x)+b_{\ell}\sin(\ell x).
#
# Here, :math:`a_{\ell}` and :math:`b_{\ell}` are the Fourier coefficients and we only
# restrict ourselves to positive frequencies :math:`\Omega_{\ell}=\ell` as :math:`E(x)` is
# real-valued. This is because :math:`B` is Hermitian, which implies (anti-)symmetry for the
# real (imaginary) Fourier coefficients.
# This is true for any number of qubits (and therefore ``RZ`` gates) we use.
#
# Using our function ``make_cost`` from above, we create the functions for several
# numbers of qubits and store both the function and its evaluations on the plotting range
# ``X``.


# Qubit numbers
Ns = [1, 2, 4, 5]
# Fix a random seed
seed = 7658741

cost_functions = []
evaluated_cost = []
for N in Ns:
    # Generate the cost function for N qubits and evaluate it
    cost = make_cost(N, seed)
    evaluated_cost.append([cost(x) for x in X])
    cost_functions.append(cost)


###############################################################################
# Let's take a look at the created :math:`E(x)` for the various numbers of qubits:


# Figure with multiple axes
fig, axs = plt.subplots(1, len(Ns), figsize=(12, 2))
for ax, N, E in zip(axs, Ns, evaluated_cost):
    ax.plot(X, E, color=green)
    # Axis and plot labels
    ax.set_title(f"{N} qubits")
    ax.set_xlabel("$x$")
_ = axs[0].set_ylabel("$E$")


###############################################################################
# Indeed we see that :math:`E(x)` is a periodic function whose complexity grows when increasing
# the number of gates parametrized by :math:`x`.
# We take a look at the frequencies that are supported by the functions using features from
# PennyLane's :mod:`~.pennylane.fourier` module.


from pennylane.fourier.visualize import bar

fig, axs = plt.subplots(2, len(Ns), figsize=(12, 4.5))
for i, cost_function in enumerate(cost_functions):
    # Compute the Fourier coefficients for N+2 frequencies
    coeffs = qml.fourier.coefficients(cost_function, 1, Ns[i] + 2)
    # Show the Fourier coefficients
    bar(coeffs, 1, axs[:, i], show_freqs=True, colour_dict={"real": green, "imag": orange})
    axs[0, i].set_title(f"{Ns[i]} qubits")
    # Set x-axis labels
    axs[1, i].text(Ns[i] + 2, axs[1, i].get_ylim()[0], f"Frequency", ha="center", va="top")
    # Clean up y-axis labels
    if i == 0:
        _ = [axs[j, i].set_ylabel(lab) for j, lab in enumerate(["$a_\ell/2$", "$b_\ell/2$"])]
    else:
        _ = [axs[j, i].set_ylabel("") for j in [0, 1]]


###############################################################################
# We find the real (imaginary) Fourier coefficients to be (anti-)symmetric as expected and
# the number of frequencies that appear in :math:`E(x)` is the same as the
# number of ``RZ`` gates we used in the circuit.
#
# The latter can be understood in the following way:
# Let's look at our parametrized unitary :math:`U(x)` a bit closer. Note that the generators of
# the used Pauli rotations commute, and that we therefore can rewrite :math:`U` as
#
# .. math ::
#
#   U(x)=\exp\left(-i x \sum_{a=1}^N \frac{1}{2}Z_a\right),
#
# i.e. the layer of rotations is *generated* by the Hermitian operator
#
# .. math ::
#
#   G = -\sum_{a=1}^N \frac{1}{2}Z_a.
#
# As :math:`Z` has the eigenvalues :math:`\pm 1` and the operators in :math:`G` act on distinct
# wires, the eigenvalue spectrum of :math:`G` is
#
# .. math ::
#
#   \{\omega_j\} = \left\{-\frac{N}{2},-\frac{N-2}{2},\dots, \frac{N-2}{2}, \frac{N}{2} \right\},
#
# which we can check in the code (we only compute the diagonal of the generator).


N = 6
# Identity matrix acting on k qubits
_eye = lambda k: np.ones(2 ** k)
# PauliZ operator acting on the ith of N qubits
PauliZ = lambda i: np.kron(_eye(i), np.kron(np.array([1, -1]), _eye(N - 1 - i)))
# Sum the PauliZ operators to the generator
generator = 0.5 * np.sum(np.array(list(map(PauliZ, range(N)))), axis=0)
# Show the unique eigenvalues
omegas = sorted(list(set(generator.astype(int))))
print(f"The eigenvalues are: {omegas}")


###############################################################################
# The frequencies in the Fourier series will be the *unique, positive* differences of these
# eigenvalues, which are:
#
# .. math ::
#
#   \{\Omega_\ell\} = \{1, 2,\dots, N\},
#
# as well as a zero frequency leading to the constant term :math:`a_0` in :math:`E(x)`.
# This is exactly what we saw in the plots above and again we also can compute those in code
# (recall from above that we don't need to count negative frequencies):


from pennylane.utils import _flatten

# Compute all differences, including duplicates
differences = _flatten(
    [[om1 - om2 for om1 in omegas if om1 >= om2] for om2 in omegas]
)
# Remove duplicates and sort in ascending order
Omegas = sorted(list(set(differences)))

print(f"The frequencies are: {Omegas}")


###############################################################################
# In general we will call the number of these frequencies :math:`R`, so that we have :math:`R=N`
# for the layer of ``RZ`` gates above.
#
# Determining the full dependence on :math:`x`
# --------------------------------------------
#
# Here we will implement the full function reconstruction of :math:`E(x)`.
# The key idea is simple: Since :math:`E(x)` is periodic with known integer frequencies, we can
# reconstruct it *exactly* by using
# `trigonometric interpolation <https://en.wikipedia.org/wiki/Trigonometric_interpolation>`_.
# We will show this both with equidistant and random shifts, corresponding to a
# `uniform <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`_ and a
# `non-uniform <https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform>`_
# discrete Fourier transform (DFT), respectively.
# We start with the equidistant case for which we can directly implement the trigonometric
# interpolation:
#
# .. math ::
#
#   x_\mu &= \frac{2\mu\pi}{2R+1}\\
#   E(x) &=\sum_{\mu=-R}^R E\left(x_\mu\right) \frac{\sin\left(\frac{2R+1}{2}(x-x_\mu)\right)} {(2R+1)\sin \left(\frac{1}{2} (x-x_\mu)\right)}\\
#   &=\sum_{\mu=-R}^R E\left(x_\mu\right) \frac{\operatorname{sinc}\left(\frac{2R+1}{2}(x-x_\mu)\right)} {\operatorname{sinc} \left(\frac{1}{2} (x-x_\mu)\right)},
#
# where we reformulated :math:`E` in the second expression using the sinc function
# (:math:`\operatorname{sinc}(x)=\sin(x)/x`) to enhance the numerical stability.
# Note that we have to take care of a rescaling factor of :math:`\pi` between this definition of
# :math:`\operatorname{sinc}` and the NumPy implementation ``np.sinc``.


sinc = lambda x: np.sinc(x / np.pi)


def full_reconstruction_equ(fun, R):
    """Reconstruct a univariate function with up to R frequencies using equidistant shifts."""
    # Shift angles for the reconstruction
    shifts = [2 * mu * np.pi / (2 * R + 1) for mu in range(-R, R + 1)]
    # Shifted function evaluations
    evals = np.array([fun(shift) for shift in shifts])

    @jax.jit
    def reconstruction(x):
        f"""Univariate reconstruction with up to {R} frequencies using equidistant shifts."""
        kernels = np.array(
            [sinc((R + 0.5) * (x - shift)) / sinc(0.5 * (x - shift)) for shift in shifts]
        )
        return np.dot(evals, kernels)

    return reconstruction


###############################################################################
# Let's see how this reconstruction is doing. We will plot it along with the original function
# :math:`E`, mark the shifted evaluation points (with crosses), and also show its deviation from
# :math:`E(x)` (lower plots).
# We will write a function for the whole procedure of comparing the functions and reuse it
# further below. For convenience, showing the deviation will be an optional feature controled by
# the ``show_diff`` keyword argument.


def compare_functions(originals, reconstructions, Ns, shifts, show_diff=True):
    """Plot two sets of functions next to each other and show their difference (in pairs)."""
    # Prepare the axes; we need fewer axes if we don't show the deviations
    if show_diff:
        fig, axs = plt.subplots(2, len(originals), figsize=(12, 4.5))
    else:
        fig, axs = plt.subplots(1, len(originals), figsize=(12, 2))
    _axs = axs[0] if show_diff else axs

    # Run over the functions and reconstructions
    for i, (orig, recon, N, _shifts) in enumerate(zip(originals, reconstructions, Ns, shifts)):
        # Evaluate the original function and its reconstruction over the plotting range
        E = np.array(list(map(orig, X)))
        E_rec = np.array(list(map(recon, X)))
        # Evaluate the original function at the positions used in the reconstruction
        E_shifts = np.array(list(map(orig, _shifts)))

        # Show E, the reconstruction, and the shifts (top axes)
        _axs[i].plot(X, E, lw=2, color=orange)
        _axs[i].plot(X, E_rec, linestyle=":", lw=3, color=green)
        _axs[i].plot(_shifts, E_shifts, ls="", marker="x", c=red)
        # Manage plot titles and xticks
        _axs[i].set_title(f"{N} qubits")

        if show_diff:
            # [Optional] Show the reconstruction deviation (bottom axes)
            axs[1, i].plot(X, E - E_rec, color=blue)
            axs[1, i].set_xlabel("$x$")
            # Hide the xticks of the top x-axes if we use the bottom axes
            _axs[i].set_xticks([])

    # Manage y-axis labels
    _ = _axs[0].set_ylabel("$E$")
    if show_diff:
        _ = axs[1, 0].set_ylabel("$E-E_{rec}$")

    return axs


reconstructions_equ = list(map(full_reconstruction_equ, cost_functions, Ns))
equ_shifts = [[2 * mu * np.pi / (2 * N + 1) for mu in range(-N, N + 1)] for N in Ns]
axs = compare_functions(cost_functions, reconstructions_equ, Ns, equ_shifts)


###############################################################################
# *It Works!*
#
# Now let's test the reconstruction with less regular sampling points on which to evaluate
# :math:`E`. This means we can no longer use the closed from expression from above but switch
# to solving the set of equations
#
# .. math ::
#
#   E(x_\mu) = a_0 + \sum_{\ell=1}^R a_{\ell}\cos(\ell x_\mu)+b_{\ell}\sin(\ell x_\mu)
#
# with the --- now irregular --- sampling points :math:`x_\mu`.
# For this, we set up the matrix
#
# .. math ::
#
#   C_{\mu\ell} = \begin{cases}
#   1 &\text{ if } \ell=0\\
#   \cos(\ell x_\mu) &\text{ if } 1\leq\ell\leq R\\
#   \sin(\ell x_\mu) &\text{ if } R<\ell\leq 2R,
#   \end{cases}
#
# collect the Fourier coefficients of :math:`E` into the vector
# :math:`\boldsymbol{W}=(a_0, \boldsymbol{a}, \boldsymbol{b})`, and the evaluations of :math:`E`
# into another vector called :math:`\boldsymbol{E}` so that
#
# .. math ::
#
#   \boldsymbol{E} = C \boldsymbol{W} \Rightarrow \boldsymbol{W} = C^{-1}\boldsymbol{E}.
#
# Let's implement this right away! We will take the function and the shifts :math:`x_\mu` as
# inputs, inferring :math:`R` from the number of the provided shifts, which is :math:`2R+1`.


def full_reconstruction_gen(fun, shifts):
    """Reconstruct a univariate trigonometric function using arbitrary shifts."""
    R = (len(shifts) - 1) // 2
    frequencies = np.array(list(range(1, R + 1)))

    # Construct the matrix C case by case
    C1 = np.ones((2 * R + 1, 1))
    C2 = np.cos(np.outer(shifts, frequencies))
    C3 = np.sin(np.outer(shifts, frequencies))
    C = np.hstack([C1, C2, C3])

    # Evaluate the function to reconstruct at the shifted positions
    evals = np.array(list(map(fun, shifts)))

    # Solve the system of linear equations by inverting C
    W = np.linalg.inv(C) @ evals

    # Extract the Fourier coefficients
    a0 = W[0]
    a = W[1 : R + 1]
    b = W[R + 1 :]

    # Construct the Fourier series
    @jax.jit
    def reconstruction(x):
        f"""Univariate reconstruction with up to {R} frequencies based on arbitrary shifts."""
        return a0 + np.dot(a, np.cos(frequencies * x)) + np.dot(b, np.sin(frequencies * x))

    return reconstruction


###############################################################################
# Again, let's see the reconstruction in action:
# We will sample the shifts :math:`x_\mu` at random in :math:`[-\pi,\pi)`.


shifts = [rnd.random(2 * N + 1) * 2 * np.pi - np.pi for N in Ns]
reconstructions_gen = list(map(full_reconstruction_gen, cost_functions, shifts))
axs = compare_functions(cost_functions, reconstructions_gen, Ns, shifts)


###############################################################################
# Again, we obtain a perfect reconstruction of :math:`E(x)` up to numerical errors.
# We see that the deviation from the original cost function became larger than for equidistant
# shifts for some of the qubit numbers but it still remains much smaller than any energy scale of
# relevance in applications.
# The reason for these larger deviations are evaluation positions :math:`x_\mu` that were sampled
# very close to each other, so that inverting the matrix :math:`C` becomes less stable numerically.
# Conceptually, we see that the reconstruction does not rely on equidistant evaluations points.
#
# .. note ::
#
#     For some applications, the number of frequencies :math:`R` is not known exactly but an upper
#     bound for :math:`R` might be available. In this case, it is very useful that a reconstruction
#     that assumes *too many* frequencies in :math:`E(x)` works perfectly fine.
#     However, it has the disadvantage of spending too many evaluations on the reconstruction
#     and similarly the number of required measurements, which is meaningful for the (time)
#     complexity of quantum algorithms, does so as well!
#
# Rotosolve
# ---------
#
# .. note ::
#
#     Before going through this section, we recommend that readers refer to the
#     :doc:`Rotosolve & Rotoselect tutorial </demos/tutorial_rotoselect>`.
#
# Now what can we do with these reconstruction methods? Before diving into the computation of
# derivatives, a first idea is to obtain the global minimum of this univariate function.
# This can be done via `convex optimization <https://en.wikipedia.org/wiki/Convex_optimization>`_
# (see Theorem 7 in [#CalcPQC]_), or with a global
# optimization technique, which is feasible because we look at a function of a single parameter.
# Two such global optimization techniques available via ``scipy.optimize`` are |brute|_
# (a simple `grid search <https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search>`_
# algorithm) and |shgo|_ (simplicial homology global optimization).
# For this step we will use the equidistant shifts from the first reconstruction above.


from scipy.optimize import brute, shgo

optimizer = "brute"  # Can be changed to "shgo"
axs = compare_functions(cost_functions, reconstructions_equ, Ns, equ_shifts, show_diff=False)

for i, (recon, N) in enumerate(zip(reconstructions_equ, Ns)):
    # Minimize the (classical) reconstructed function
    if optimizer == "brute":
        x_min, y_min, *_ = brute(recon, ranges=(xlim,), Ns=100, full_output=True)
    elif optimizer == "shgo":
        opt_res = shgo(recon, (xlim,))
        x_min, y_min = opt_res.x, opt_res.fun
    # Show the obtained minimum
    axs[i].plot([x_min], [y_min], marker="o", color=pink)


###############################################################################
# The idea of using this one-dimensional minimization for functions of multiple variables, by
# applying it coordinate-wise, was proposed in general in [#CalcPQC]_ and for Pauli rotation
# gates in particular in [#Rotosolve]_. The name for this algorithm, put forward by the latter
# paper, is *Rotosolve* and methods that optimize a subset of parameters at a
# time are often referred to as *layerwise training*.
# For completeness and as a minimal example, we here write out Rotosolve for quantum gates with
# equidistant frequencies, but PennyLane also provides a full implementation via |Rotosolve_code|_.


def rotosolve_substep(univariate_fun, R, gridsearch_steps):
    """Globally minimize a univariate function using ``scipy.optimize.brute``."""
    recon = full_reconstruction_equ(univariate_fun, R)
    center, width = 0.0, 2 * np.pi
    # Refine the gridsearch multiple times
    for _ in range(gridsearch_steps):
        ranges = ((center - width / 2, center + width / 2),)
        center, y_center, *_ = brute(recon, ranges=ranges, Ns=100, full_output=True)
        width /= 100
    return center, y_center


def rotosolve_step(fun, param, Rs, gridsearch_steps=1):
    """Update all parameters once by restricting a function to one parameter at a time."""
    # Cache minima of reconstructed functions
    y_values = []
    # Canonical unit vectors
    vecs = np.eye(len(param))
    for vec, R in zip(vecs, Rs):
        # Restrict fun to current coordinate axis
        univariate_fun = lambda x: fun(param + x * vec)
        # Reconstruct univariate_fun and minimize reconstruction
        x_min, y_min = rotosolve_substep(univariate_fun, R, gridsearch_steps)
        # Update the current parameter
        param += x_min * vec
        # Cache minima of reconstructed functions
        y_values.append(y_min)
    return param, y_values


###############################################################################
# We can test this on a small toy model, using a random initial state and observable as before
# but including multiple layers of parametrized gates, with one parameter per layer. Let's
# first define the cost function:


ops = [qml.RX, qml.RY, qml.RZ]
Rs = [4, 2, 3]
N = max(Rs)
dev = qml.device("default.qubit.jax", wires=N)


@jax.jit
@qml.qnode(dev, interface="jax")
def cost(param):
    """Multivariate cost function with various numbers of frequencies per parameter."""
    qml.QubitStateVector(random_state(N, seed), wires=dev.wires)
    for j, (par, R, op) in enumerate(zip(param, Rs, ops)):
        for w in dev.wires[:R]:
            op(par, wires=w)
        if j < len(Rs) - 1:
            for i in range(N):
                qml.CNOT(wires=[dev.wires[i], dev.wires[(i + 1) % N]])
    return qml.expval(qml.Hermitian(random_observable(N, seed), wires=dev.wires))


###############################################################################
# Now we sample initial parameters and run the ``rotosolve_step`` function repeatedly,
# say 5 times corresponding to 15 univariate minimizations, and observe the cost
# being minimized.


rnd.seed(seed)
# Initial parameters and cost
param = rnd.random(len(Rs)) * 2 * np.pi - np.pi
y_values = [cost(param)]
print(f"Initial cost: {y_values[0]}")
# 5 iterations of Rotosolve, each updating all parameters once
for step in range(5):
    param, _y_values = rotosolve_step(cost, param, Rs, gridsearch_steps=2)
    print(f"Optimization substeps during step {step+1}:\n{_y_values}")
    # Store intermediate cost values
    y_values.extend(_y_values)
print(f"Final cost: {y_values[-1]}")


###############################################################################
# Let's also look at the optimization in a plot. We will show the energy after each
# ``rotosolve_substep`` and additionally mark the energies after each ``rotosolve_step``,
# between which each parameter was updated once (orange circles).


# Number of substeps at which steps are completed
iterations = range(0, len(y_values), len(Rs))
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# Energies after substeps
ax.plot(y_values, color=green)
# Energies after steps
ax.plot(iterations, y_values[:: len(Rs)], ls="", marker="o", color=orange)
# Labels
ax.set_xlabel("$\#$ Univariate updates")
_ = ax.set_ylabel("$E$")


###############################################################################
#
# Generalized parameter-shift rules
# ---------------------------------
#
# Next, we use a modified reconstruction strategy that only obtains the odd or even part of
# :math:`E(x)` to compute odd- and even-order derivatives. While this method yields the *entire*
# odd or even functional dependence (see below), we here only extract the first and second
# derivative directly, based on the analytical form of the odd and even part, respectively.
# We start with the first derivative, which certainly is used the most:
#
# .. math ::
#
#   E'(0) = \sum_{\mu=1}^{2R} E\left(\frac{2\mu-1}{2R}\pi\right) \frac{(-1)^{\mu-1}}{4R\sin^2\left(\frac{2\mu-1}{4R}\pi\right)},
#
# This is straight-forward to implement by defining the coefficients and evaluating
# :math:`E` at the shifted positions :math:`x_\mu`:


def parameter_shift_first(fun, R):
    """Compute the first-order derivative of a function with R frequencies at 0."""
    shifts = [(2 * mu - 1) * np.pi / (4 * R) for mu in range(1, 2 * R + 1)]
    # Classically computed coefficients
    coeffs = np.array(
        [(-1) ** mu / (4 * R * np.sin(shift) ** 2) for mu, shift in enumerate(shifts)]
    )
    # Evaluations of the cost function E(x_mu)
    evaluations = np.array([fun(2 * shift) for shift in shifts])
    # Contract coefficients with evaluations
    return np.dot(coeffs, evaluations)


###############################################################################
# The second-order derivative takes a similar form but we have to take care of the evaluation at
# :math:`0` and the corresponding coefficient, separately:
#
# .. math ::
#
#   E''(0) = -E(0)\frac{2R^2+1}{6} - \sum_{\mu=1}^{2R-1} E\left(\frac{\mu\pi}{R}\right)\frac{(-1)^\mu}{2\sin^2 \left(\frac{\mu\pi}{2R}\right)}.
#
# Let's code this up, again we only get slight complications from the special evaluation
# at :math:`0`:


def parameter_shift_second(fun, R):
    """Compute the second-order derivative of a function with R frequencies at 0."""
    shifts = [mu * np.pi / (2 * R) for mu in range(1, 2 * R)]
    # Classically computed coefficients for the main sum
    _coeffs = [(-1) ** mu / (2 * np.sin(shift) ** 2) for mu, shift in enumerate(shifts)]
    # Include the coefficients for the "special" term E(0).
    coeffs = np.array([-(2 * R ** 2 + 1) / 6] + _coeffs)
    # Evaluate at the regularily shifted positions
    _evaluations = [fun(2 * shift) for shift in shifts]
    # Include the "special" term E(0).
    evaluations = np.array([fun(0)] + _evaluations)
    # Contract coefficients with evaluations.
    return np.dot(coeffs, evaluations)


###############################################################################
# Let's compare these two shift rules to the finite-difference derivative commonly used for
# numerical differentiation. We choose a finite difference of :math:`d_x=5\times 10^{-5}`.


# Compute the parameter-shift derivatives
ps_der1 = list(map(parameter_shift_first, cost_functions, Ns))
ps_der2 = list(map(parameter_shift_second, cost_functions, Ns))

# Compute the finite-difference derivatives
dx = 5e-5
fd_der1 = [(orig(dx / 2) - orig(-dx / 2)) / (dx) for orig, N in zip(cost_functions, Ns)]
fd_der2 = [
    ((orig(dx) - orig(0)) / dx - (orig(0) - orig(-dx)) / dx) / dx
    for orig, N in zip(cost_functions, Ns)
]

# Compare derivatives
print("Number of qubits/RZ gates:         ", *Ns, sep=" " * 9)
print(f"First-order parameter-shift rule:  {np.round(np.array(ps_der1), 6)}")
print(f"First-order finite difference:     {np.round(np.array(fd_der1), 6)}")
print(f"Second-order parameter-shift rule: {np.round(np.array(ps_der2), 6)}")
print(f"Second-order finite difference:    {np.round(np.array(fd_der2), 6)}")


###############################################################################
# The derivatives coincide, great!
#
# .. note ::
#
#     While we used the :math:`2R+1` evaluations :math:`x_\mu=\frac{2\mu\pi}{2R+1}` for the full
#     reconstruction, both derivatives only require :math:`2R` calls to the respective circuit.
#     Note that the derivatives can be computed at any position :math:`x_0` other than :math:`0`
#     by simply reconstructing the function :math:`E(x+x_0)`, which will have the same functional
#     form as :math:`E(x)`.
#
# Automatically differentiated reconstructions
# --------------------------------------------
#
# Above we used explicit parameter-shift rule formulas derived manually from the analytical form
# of the odd and even function reconstruction.
# However, we can also implement this reconstruction method as a function; using PennyLane's
# automatic differentiation backends, this then enables us to obtain the derivatives at the point
# of interest (For odd-order derivatives, we use the reconstruction of the odd part, for the
# even-order derivatives that of the even part).
#
# Here is how to do this, using modified Dirichlet kernels and equidistant shifts. For the odd
# reconstruction we have
#
# .. math ::
#
#   E_\text{odd}(x) &= \sum_{\mu=1}^R E_\text{odd}(x_\mu) \tilde{D}_\mu(x)\\
#   \tilde{D}_\mu(x) &= \frac{\sin(R (x-x_\mu))}{2R \tan\left(\frac{1}{2} (x-x_\mu)\right)} - \frac{\sin(R (x+x_\mu))}{2R \tan\left(\frac{1}{2} (x+x_\mu)\right)},
#
# which we can implement using the reformulation
#
# .. math ::
#
#   \frac{\sin(X)}{\tan(Y)}=\frac{X}{Y}\frac{\operatorname{sinc}(X)}{\operatorname{sinc}(Y)}\cos(Y)
#
# for the kernel.


shifts_odd = lambda R: [(2 * mu - 1) * np.pi / (2 * R) for mu in range(1, R + 1)]
# Odd linear combination of Dirichlet kernels
D_odd = lambda x, R: np.array(
    [
        (
            sinc(R * (x - shift)) / sinc(0.5 * (x - shift)) * np.cos(0.5 * (x - shift))
            - sinc(R * (x + shift)) / sinc(0.5 * (x + shift)) * np.cos(0.5 * (x + shift))
        )
        for shift in shifts_odd(R)
    ]
)


def odd_reconstruction_equ(fun, R):
    """Reconstruct the odd part of an ``R``-frequency input function via equidistant shifts."""
    evaluations = np.array([(fun(shift) - fun(-shift)) / 2 for shift in shifts_odd(R)])

    def reconstruction(x):
        f"""Odd reconstruction with {R} frequencies based on equidistant shifts."""
        return np.dot(evaluations, D_odd(x, R))

    return reconstruction


odd_reconstructions = list(map(odd_reconstruction_equ, cost_functions, Ns))


###############################################################################
# The even part on the other hand takes the form
#
# .. math ::
#
#   E_\text{even}(x) &= \sum_{\mu=0}^R E_\text{even}(x_\mu) \hat{D}_\mu(x)\\
#   \hat{D}_\mu(x) &=
#   \begin{cases}
#      \frac{\sin(Rx)}{2R \tan(x/2)} &\text{if } \mu = 0 \\
#      \frac{\sin(R (x-x_\mu))}{2R \tan\left(\frac{1}{2} (x-x_\mu)\right)} + \frac{\sin(R (x+x_\mu))}{2R \tan\left(\frac{1}{2} (x+x_\mu)\right)} & \text{if } \mu \in [R-1] \\
#      \frac{\sin(R (x-\pi))}{2R \tan\left(\frac{1}{2} (x-\pi)\right)} & \text{if } \mu = R.
#   \end{cases}
#
# Note that the shifted positions :math:`\{x_\mu\}` differ between the odd and even case.
# We also set up a function that performs both partial reconstructions and sums the resulting
# functions to the full Fourier series.


shifts_even = lambda R: [mu * np.pi / R for mu in range(1, R)]
# Even linear combination of Dirichlet kernels
D_even = lambda x, R: np.array(
    [
        (
            sinc(R * (x - shift)) / sinc(0.5 * (x - shift)) * np.cos(0.5 * (x - shift))
            + sinc(R * (x + shift)) / sinc(0.5 * (x + shift)) * np.cos(0.5 * (x + shift))
        )
        for shift in shifts_even(R)
    ]
)
# Special cases of even kernels
D0 = lambda x, R: sinc(R * x) / (sinc(x / 2)) * np.cos(x / 2)
Dpi = lambda x, R: sinc(R * (x - np.pi)) / sinc((x - np.pi) / 2) * np.cos((x - np.pi) / 2)


def even_reconstruction_equ(fun, R):
    """Reconstruct the even part of ``R``-frequency input function via equidistant shifts."""
    _evaluations = np.array([(fun(shift) + fun(-shift)) / 2 for shift in shifts_even(R)])
    evaluations = np.array([fun(0), *_evaluations, fun(np.pi)])
    kernels = lambda x: np.array([D0(x, R), *D_even(x, R), Dpi(x, R)])

    def reconstruction(x):
        f"""Even reconstruction with {R} frequencies based on equidistant shifts."""
        return np.dot(evaluations, kernels(x))

    return reconstruction


even_reconstructions = list(map(even_reconstruction_equ, cost_functions, Ns))


def summed_reconstruction_equ(fun, R):
    """Sum an odd and an even reconstruction into the full function."""
    _odd_part = odd_reconstruction_equ(fun, R)
    _even_part = even_reconstruction_equ(fun, R)

    def reconstruction(x):
        f"""Full function with {R} frequencies based on separate odd/even reconstructions."""
        return _odd_part(x) + _even_part(x)

    return reconstruction


summed_reconstructions = list(map(summed_reconstruction_equ, cost_functions, Ns))


###############################################################################
# Let's now look at these even (blue) and odd (red) reconstructions and how they indeed
# combine to the full function (we will use the ``compare_functions`` utility from above
# for the latter).


# Obtain the shifts for the reconstruction of both parts
odd_and_even_shifts = [
    (
        shifts_odd(R)
        + shifts_even(R)
        + list(-1 * np.array(shifts_odd(R)))
        + list(-1 * np.array(shifts_odd(R)))
        + [0, np.pi]
    )
    for R in Ns
]

# Show the reconstructed parts and the sums
axs = compare_functions(cost_functions, summed_reconstructions, Ns, odd_and_even_shifts)
for i, (odd_recon, even_recon) in enumerate(zip(odd_reconstructions, even_reconstructions)):
    E_odd = np.array(list(map(odd_recon, X)))
    E_even = np.array(list(map(even_recon, X)))
    axs[0, i].plot(X, E_odd, color=red)
    axs[0, i].plot(X, E_even, color=blue)
_ = axs[1, 0].set_ylabel("$E-(E_{odd}+E_{even})$")


###############################################################################
# Great! The even and odd part indeed combine into the correct function again. But what did we
# gain? Nothing, actually, for the full reconstruction! Quite the opposite, we spent :math:`2R`
# evaluations of :math:`E` on each part, that is :math:`4R` evaluations overall to obtain a
# description of the full function :math:`E`, instead of the :math:`2R+1` evaluations from the
# first approach.
# However, sometimes we might just be interested in the odd or even part of :math:`E` alone,
# for example to compute odd- or even-order derivatives at a chosen point (with respect to which
# the two parts then have to be odd/even).
# Using autodifferentiation as mentioned above, in particular JAX, we can compute higher-order
# derivatives without precomputing the corresponding shift rules by hand:


# An iterative function computing the ``order``th derivative of a function ``f`` with JAX
grad_gen = lambda f, order: grad_gen(jax.grad(f), order - 1) if order > 0 else f

# Compute the first, second, and fifth derivative
for order, name in zip([1, 2, 4], ["First", "Second", "4th"]):
    recons = odd_reconstructions if order % 2 else even_reconstructions
    recon_name = "odd " if order % 2 else "even"
    cost_grads = [grad_gen(orig, order)(0.0) for orig in cost_functions]
    recon_grads = [grad_gen(recon, order)(0.0) for recon in recons]
    all_equal = (
        "All entries match" if np.allclose(cost_grads, recon_grads) else "Some entries differ!"
    )
    print(f"{name} derivatives via jax: {all_equal}")
    print("From the cost functions:       ", np.round(np.array(cost_grads), 6))
    print(f"From the {recon_name} reconstructions: ", np.round(np.array(recon_grads), 6), "\n")


###############################################################################
# And this is all we want to show here about univariate function reconstructions and generalized
# parameter shift rules.
# Note that the techniques above can partially be extended to frequencies that are not
# integer-valued, but many closed form expressions are no longer valid.
# For the reconstruction, the approach via Dirichlet kernels does no longer work in the general
# case and instead a system of equations has to be solved, but with generalized
# frequencies :math:`\{\Omega_\ell\}` instead of :math:`\{\ell\}`.
#
#
# References
# ----------
#
# .. [#GenPar]
#
#     David Wierichs, Josh Izaac, Cody Wang, Cedric Yen-Yu Lin.
#     "General parameter-shift rules for quantum gradients".
#     `arXiv preprint arXiv:2107.12390 <https://arxiv.org/abs/2107.12390>`__.
#
# .. [#CalcPQC]
#
#     Javier Gil Vidal, Dirk Oliver Theis. "Calculus on parameterized quantum circuits".
#     `arXiv preprint arXiv:1812.06323 <https://arxiv.org/abs/1812.06323>`__.
#
# .. [#Rotosolve]
#
#     Mateusz Ostaszewski, Edward Grant, Marcello Benedetti.
#     "Structure optimization for parameterized quantum circuits".
#     `arXiv preprint arXiv:1905.09692 <https://arxiv.org/abs/1905.09692>`__.
#
# .. [#AlgeShift]
#
#     Artur F. Izmaylov, Robert A. Lang, Tzu-Ching Yen.
#     "Analytic gradients in variational quantum algorithms: Algebraic extensions of the parameter-shift rule to general unitary transformations".
#     `arXiv preprint arXiv:2107.08131 <https://arxiv.org/abs/2107.08131>`__.
#
# .. [#GenDiffRules]
#
#     Oleksandr Kyriienko, Vincent E. Elfving.
#     "Generalized quantum circuit differentiation rules".
#     `arXiv preprint arXiv:2108.01218 <https://arxiv.org/abs/2108.01218>`__.
#
# .. |brute| replace:: ``brute``
# .. _brute: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html
#
# .. |shgo| replace:: ``shgo``
# .. _shgo: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html
#
# .. |Rotosolve_code| replace:: ``qml.RotosolveOptimizer``
# .. _Rotosolve_code: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.RotosolveOptimizer.html
