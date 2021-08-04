r"""

.. _general_parshift:

General parameter-shift rules for quantum gradients
===================================================

.. meta::
    :property="og:description": Reconstruct quantum functions and compute their derivatives.
    :property="og:image": https://pennylane.ai/qml/_images/flowchart.png

.. related::

   tutorial_rotoselect Leveraging trigonometry with Rotoselect
   tutorial_quantum_analytic_descent Building multivariate models with QAD


*Author: David Wierichs (Xanadu resident). Posted: ?? August 2021.*

todo: INTRO

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
    states = unitary_group.rvs(2**N, random_state=rnd.default_rng(seed))
    return states[0]

def random_observable(N, seed):
    rnd.seed(seed)
    # Generate real and imaginary part separately and (anti-)symmetrize them for Hermiticity
    real_part, imag_part = rnd.random((2, 2**N, 2**N))
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
from jax import numpy as np

# Choose number of qubits
N = 4
# Test random state
psi = random_state(N, 1234)
print("psi is normalized:       ", np.isclose(np.linalg.norm(psi), 1.0))
print("psi has shape (2**N,):   ", psi.shape==(2**N,))
# Test random observable
B = random_observable(N, 1234)
print("B is Hermitian:          ", np.allclose(B, B.T.conj()))
print("B has shape (2**N, 2**N):", B.shape==(2**N, 2**N))


###############################################################################
# Now let's set up a "circuit generator", namely a function that will create a ``device`` and the
# ``cost`` function using :math:`|\psi\rangle` as initial state and measuring :math:`B`, depending
# on the number of qubits:


import pennylane as qml

def device_and_cost(N, seed):
    dev = qml.device('default.qubit', wires=N)

    @qml.qnode(dev, interface='jax')
    def cost(x):
        qml.QubitStateVector(random_state(N, seed), wires=dev.wires)
        for w in dev.wires:
            qml.RZ(x, wires=w)
        return qml.expval(qml.Hermitian(random_observable(N, seed), wires=dev.wires))

    return dev, cost

###############################################################################
# Section II A of [#GenPar]_ shows that the output of this circuit, namely the function
# .. math ::
#
#   E(x)=\langle\psi | U^\dagger(x) B U(x)|\psi\rangle
#
# with :math:`U(x)` summarizing the ``RZ`` gates, takes the form of a Fourier series in the
# variable :math:`x`.
# This is true for any number of qubits (and therefore ``RZ`` gates) we use.
# Let's take a look at :math:`E(x)` for one to five qubits (and store those functions for later):


import matplotlib.pyplot as plt
import copy

# Qubit numbers
Ns = list(range(1, 6))
# Fix a random seed
seed = 7658741
# Set the plotting range on the x-axis
xlim = (-np.pi, np.pi)
fig, axs = plt.subplots(1, len(Ns), figsize=(16,2))
cost_functions = []
X = np.linspace(*xlim, 60)
for ax, N in zip(axs, Ns):
    # Generate the cost function for N qubits and plot it
    dev, cost = device_and_cost(N, seed)
    ax.plot(X, [cost(x) for x in X])
    ax.set_title(f"{N} qubits")
    ax.set_xlabel('$x$')
    # Store the cost function for later
    cost_functions.append(copy.deepcopy(cost))
axs[0].set_ylabel('$E$');


###############################################################################
# Indeed we see that :math:`E(x)` is a periodic function whose complexity grows when increasing
# the number of gates parametrized by :math:`x`.
# We take a look at the frequencies that are supported by the functions using features from
# PennyLane's ``fourier`` module.


from pennylane.fourier.visualize import bar

fig, axs = plt.subplots(2, len(Ns), figsize=(16, 4.5))
for i, cost_function in enumerate(cost_functions):
    # Compute the Fourier coefficients for 5 frequencies
    coeffs = qml.fourier.coefficients(cost_function, 1, 5)
    # Show the Fourier coefficients
    bar(coeffs, 1, axs[:, i], show_freqs=True)
    axs[0, i].set_title(f"{Ns[i]} qubits")
    # Set x-axis labels
    axs[1, i].text(5, axs[1, i].get_ylim()[0], f"Frequency", ha='center', va='top')
    # Clean up y-axis labels
    if i>1:
        axs[0, i].set_ylabel('')
        axs[1, i].set_ylabel('')


###############################################################################
# We find the number of (positive) frequencies that appear in :math:`E(x)` to be the same as the
# number of ``RZ`` gates we used in the circuit.
# This is no coincidence:
#
# Let's look at our parametrized unitary
#
# .. math ::
#
#   U(x)=\prod_{a=1}^N R_Z^{(a)}(x) = \prod_{a=1}^N \exp\left(-i\frac{x}{2} Z_a\right)
#
# a bit closer. Note that the generators of the used Pauli rotations commute, and that we therefore
# can rewrite :math:`U` as
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
#   \{\omega_j\} = \left\{-\frac{N}{2},-\frac{N-2}{2},\dots, \frac{N-2}{2}, \frac{N}{2} \right\}.
#
# As discussed in Sec. II A of the paper, the frequencies in the Fourier series will be the
# *unique, positive* differences of these eigenvalues, which are:
#
# .. math ::
#
#   \{\Omega_\ell\} = \{1, 2,\dots, N\},
#
# as well as a zero frequency leading to a constant term in :math:`E(x)`.
# This is exactly what we saw in the plots above.
# In general we will call the number of these frequencies :math:`R`, so that we have :math:`R=N`
# for the layer of ``RZ`` gates above.
#
# The reason we can restrict ourselves to positive frequencies is that :math:`E(x)` is real-valued.
# This is because :math:`B` is Hermitian, and it implies (anti-)symmetry for the real (imaginary)
# Fourier coefficients, as one can also see in the spectral plots above.
#
# Determining the full dependence on :math:`x`
# --------------------------------------------
#
# Here we will implement the full function reconstruction described in Sec. III A of [#GenPar]_.
# We will show it both with equidistant and random shifts, corresponding to a uniform and a
# non-uniform discrete Fourier transform (DFT), respectively.
# We start with the equidistant case, which is described in more detail in App. A2a and for which
# we can directly implement Eq. (A15):
#
# .. math ::
#
#   x_\mu &= \frac{2\mu\pi}{2R+1}\\
#   E(x) &=\sum_{\mu=-R}^R E\left(x_\mu\right) \frac{\sin\left(\frac{2R+1}{2}(x-x_\mu)\right)} {(2R+1)\sin \left(\frac{1}{2} (x-x_\mu)\right)}\\
#   &=\sum_{\mu=-R}^R E\left(x_\mu\right) \frac{\operatorname{sinc}\left(\frac{2R+1}{2}(x-x_\mu)\right)} {\operatorname{sinc} \left(\frac{1}{2} (x-x_\mu)\right)},
#
# where we reformulated :math:`E` using the sinc function (:math:`\operatorname{sinc}(x)=\sin(x)/x`)
# to enhance the numerical stability.
# Note that we have to take care of a rescaling factor of :math:`\pi` between the definition of
# :math:`\operatorname{sinc}` above and the NumPy implementation ``np.sinc``.


def full_reconstruction_equ(fun, R):
    """Reconstruct a univariate trigonometric function
    with up to R frequencies using equidistant shifts."""
    # Shift angles for the reconstruction
    shifts = [2*mu*np.pi/(2*R+1) for mu in range(-R, R+1)]
    # Shifted function evaluations
    evals = np.array([fun(shift) for shift in shifts])
    kernels = lambda x: np.array([
        sinc((R+0.5)*(x-shift)/np.pi)/sinc(0.5*(x-shift)/np.pi) for shift in shifts
    ])
    return lambda x: np.sum(evals * kernels(x))


###############################################################################
# Let's see how this reconstruction is doing. We will plot it along with the original function
# :math:`E`, mark the shifted evaluation points (with crosses), and also show its deviation from
# :math:`E(x)` (lower plots).
# We will write a function for the whole procedure of comparing the functions and reuse it
# further below.


def compare_functions(originals, reconstructions, Ns, show_diff=True):
    if show_diff:
        fig, axs = plt.subplots(2, len(originals), figsize=(16,4))
    else:
        fig, *axs = plt.subplots(1, len(originals), figsize=(16,4))
    for i, (orig, recon, N) in enumerate(zip(originals, reconstructions, Ns):
        shifts = [2*mu*np.pi/(2*N+1) for mu in range(-N, N+1)]
        E = np.array([orig(x) for x in X])
        E_rec = np.array([recon(x) for x in X])
        E_shifts = np.array([orig(shift) for shift in shifts])

        # Show E, the reconstruction, and the shifts (top)
        axs[0,i].plot(X, E)
        axs[0,i].plot(X, E_rec, linestyle='--')
        axs[0,i].plot(shifts, E_shifts, ls='', marker='x', c='red')
        axs[0,i].set_title(f"{N} qubits")
        axs[0,i].set_xticks([])
        # Show the reconstruction error (bottom)
        if show_diff:
            axs[1,i].plot(X, E-E_rec)
            axs[1,i].set_xlabel('$x$')
    axs[0,0].set_ylabel('$E$');
    if show_diff:
        axs[1,0].set_ylabel('$E-E_{rec}$');
    return axs

reconstructions_equ = [
    copy.deepcopy(full_reconstruction_equ(orig, N)) for orig, N in zip(cost_functions, Ns)
]
compare_functions(cost_functions, reconstructions_equ, Ns);

###############################################################################
# *Works.*
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
# collect the Fourier coefficients of $E$ into the vector
# :math:`\boldsymbol{w}=(a_0, \boldsymbol{a}, \boldsymbol{b})`, and the evaluations of :math:`E`
# into another vector called :math:`\boldsymbol{E}` so that
#
# .. math ::
#
#   \boldsymbol{E} = C \boldsymbol{w} \Rightarrow \boldsymbol{w} = C^{-1}\boldsymbol{E}.
#
# Let's implement this right away! We will take the function and the shifts :math:`x_\mu` as
# inputs, inferring :math:`R` from the number of the provided shifts, which is :math:`2R+1`.
#
# to do: We will sample the shifts :math:`x_\mu` at random in :math:`[-\pi,\pi)`.


def full_reconstruction_gen(fun, shifts):
    """Reconstruct a univariate trigonometric function using arbitrary shifts."""
    R = (len(shifts)-1)//2
    frequencies = np.array(list(range(1, R+1)))
    # Construct the matrix C case by case
    C1 = np.ones((2*R+1, 1))
    C2 = np.cos(np.outer(shifts, frequencies))
    C3 = np.sin(np.outer(shifts, frequencies))
    C = np.hstack([C1, C2, C3])
    # Evaluate the function to reconstruct at the shifted positions
    evals = np.array([fun(shift) for shift in shifts])
    # Solve the system of linear equations by inverting C
    w = np.linalg.inv(C) @ evals
    # Extract the Fourier coefficients
    a0 = w[0]
    a = w[1:R+1]
    b = w[R+1:]
    # Construct the Fourier series
    reconstruction = lambda x: (
        a0 + np.dot(a, np.cos(frequencies*x)) + np.dot(b, np.sin(frequencies*x))
    )
    return reconstruction


###############################################################################
# Again, let's see the reconstruction in action:

reconstructions_gen = []
for orig, N in zip(cost_functions, Ns):
    shifts = rnd.random(2*N+1)*2*np.pi-np.pi
    reconstructions_gen.append(full_reconstruction_gen(orig, shifts))
compare_functions(cost_functions, reconstructions_gen, Ns);


###############################################################################
# Again, we obtain a perfect reconstruction of :math:`E(x)` up to numerical errors.
# We see that the deviation from the original cost function became larger than for equidistant
# shifts for some of the qubit numbers but it still remains much smaller than any energy scale of
# relevance in applications.
# The reason for these larger deviations are evaluation positions :math:`x_\mu` that were sampled
# very close to each other, so that inverting the matrix :math:`C` becomes less stable numerically.
# Conceptually, we see that the reconstruction does not rely on equidistant evaluations points.
#
# Too many sampling points and redundant reconstruction
# -----------------------------------------------------
#
# For some applications, the number of frequencies :math:`R` is not known exactly but an upper
# bound might be available. In this case, it is very useful that a reconstruction which assumes
# *too many* frequencies in :math:`E(x)` works perfectly fine, and just spends too many evaluations.
# Here we demonstrate this by repeating the previous code cell, but assume :math:`R` to be larger
# than we know it to be and sample four too many shifts:


reconstructions_overp = []
for orig, N in zip(cost_functions, Ns):
    # Assume R to be too big
    R = N+2
    # Sample more shifts than required for the original function
    shifts = rnd.random(2*R+1)*2*np.pi-np.pi
    reconstructions_overp.append(full_reconstruction_gen(orig, shifts))
compare_functions(cost_functions, reconstructions_overp, Ns);


###############################################################################
# There are two disadvantages when doing this: first, we have to evaluate :math:`E` at more shifted
# positions if we only know a loose upper bound to :math:`R`; and second, when using randomly
# sampled shifts, we are more likely to sample some that are close to each other, leading to
# numerical instability.
# While the latter aspect can easily be taken care of by enforcing a minimal distance between the
# sampled shifts or choosing regular ones, there is no immediate remedy for the former, which is
# highly relevant in applications.
# In addition, not only the number of distinct circuits to evaluate grows with :math:`R`, which is
# important for simulators and cloud queueing, but also the number of required measurements, which
# is meaningful for the (time) complexity of quantum operations, does so as well!

###############################################################################
#
# Rotosolve
# ---------
#
# Now what can we do with these reconstruction methods? Before diving into the computation of
# derivatives, a first idea is to obtain the global minimum of this univariate function.
# This can be done via convex optimization (see Theorem 7 in [#CalcPQC]_), or with a global
# optimization technique, which is feasible because we look at a function of a single parameter.
# Two optimization techniques available via ``scipy.optimize`` are
# `brute <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html>`_
# (a simple grid search algorithm) and
# `shgo <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html>`_
# (simplicial homology global optimization).
# For this step we will use the equidistant shifts from the first reconstruction above.
# Choose which global optimizer to use:


from scipy.optimize import brute, shgo
optimizer = 'brute' # 'shgo'
axs = compare_functions(cost_functions, reconstructions_equ, Ns, show_diff=False);
for i, recon, N in enumerate(zip(reconstructions_equ, Ns)):
    # Minimize the (classical) reconstructed function
    if optimizer=='brute':
        x_min, y_min, *_ = brute(recon, ranges=(xlim,), Ns=100, full_output=True)
    elif optimizer=='shgo':
        opt_res = shgo(recon, (xlim,))
        x_min, y_min = opt_res.x, opt_res.fun
    axs[i].plot(x_min, y_min, marker='o')


###############################################################################
# The idea of using this one-dimensional minimization for functions of multiple variables, by
# applying it coordinate-wise, was proposed in general in [#CalcPQC]_ and for Pauli rotation
# gates in particular in [#Rotosolve]_. The name for this algorithm, put forward by the latter
# paper, is *Rotosolve* and more generally, methods that optimize a subset of parameters at a
# time are often referred to as *layerwise training*.
# For completeness and as a minimal example, we here write out Rotosolve for quantum gates with
# equidistant frequencies, but PennyLane also provides a full implementation via
# ``qml.RotosolveOptimizer``.


def rotosolve_substep(univariate_fun, R, gridsearch_steps):
    """Globally minimize a univariate function using ``scipy.optimize.brute``."""
    recon = full_reconstruction_equ(univariate_fun, R)
    center, width = 0.0, 2*np.pi
    # Refine the gridsearch multiple times
    for _ in range(gridsearch_steps):
        ranges = ((center-width/2, center+width/2),)
        center, y_center, *_ = brute(recon, ranges=ranges, Ns=100, full_output=True)
        width /= 100
    return center, y_center

def rotosolve_step(fun, param, Rs, gridsearch_steps=1):
    """Update all parameter once by restricting a function to one parameter at a time."""
    # Caching minima of reconstructed functions
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
        # Caching minima of reconstructed functions
        y_values.append(y_min)
    return param, y_values

###############################################################################
# todo: add minimal test case, think about where exactly to refer to.
# We will test this function in the notebook on multivariate optimization and will show-case its first-ever usage for QAOA in yet another notebook.
#
# Generalized parameter-shift rules
# ---------------------------------
#
# Next, we use a modified reconstruction strategy that only obtains the odd or even part of
# :math:`E(x)` to compute odd- and even-order derivatives. While this method yields the entire
# odd or even functional dependence in general (see below), we here directly make use of
# Eqs. (26,27) in [#GenPar]_ to compute the first and second derivatives:
#
# .. math ::
#
#   E'(0) &= \sum_{\substack{\mu=1\\\alpha\in\{+,-\}}}^R \alpha E\left(\alpha\frac{2\mu-1}{2R}\pi\right) \frac{(-1)^{\mu-1}}{4R\sin^2\left(\frac{2\mu-1}{4R}\pi\right)}, \\
#   E''(0) &= -E(0)\frac{2R^2+1}{6} + E(\pi)\frac{(-1)^{R-1}}{2}+ \sum_{\substack{\mu=1\\\alpha\in\{+,-\}}}^{R-1} E\left(\alpha\frac{\mu\pi}{R}\right)\frac{(-1)^{\mu-1}}{2\sin^2 \left(\frac{\mu\pi}{2R}\right)}.
#
# While we used the :math:`2R+1` evaluations :math:`x_\mu=\frac{2\mu\pi}{2R+1}` for the full
# reconstruction, both derivatives only require :math:`2R` calls to the respective circuit.
# Note that the derivatives can be computed at any position :math:`x_0` other than :math:`0`
# by simply reconstructing the function :math:`E(x+x_0)`, which will have the same functional
# form as :math:`E(x)`.


def parameter_shift(fun, R, order=1):
    if order==1:
        # Classically computed coefficients, including the sign \alpha
        coeffs = np.array([
            [
                -alpha*(-1)**mu / ( 4*R*np.sin((2*mu-1)*np.pi/(4*R))**2 )
                for mu in range(1, R+1)
            ]
            for alpha in (-1, 1)
        ])
        # Evaluations of the cost function E(x_\mu)
        evaluations = np.array([
            [fun(alpha*(2*mu-1)*np.pi/(2*R)) for mu in range(1, R+1)]
            for alpha in (-1, 1)
        ])
    elif order==2:
        # Classically computed coefficients for the regular sum
        _coeffs = np.array([
            [-(-1)**mu/(2*np.sin(mu*np.pi/(2*R))**2) for mu in range(1, R)]
            for alpha in (-1, 1)
        ])
        # Include the coefficients for the "special" terms E(0) and E(\pi)
        coeffs = np.hstack([_coeffs, np.array([[-(2*R**2+1)/6], [-(-1)**R/2]])])
        # Evaluate at the regularily shifted positions
        _evaluations = np.array([
            [fun(alpha*mu*np.pi/R) for mu in range(1, R)] for alpha in (-1, 1)
        ])
        # Include the "special" terms E(0) and E(\pi).
        evaluations = np.hstack([_evaluations, np.array([[fun(0)],[fun(np.pi)]])])
    # contract coefficients with evaluations.
    return np.sum(coeffs * evaluations)


###############################################################################
# Let's compare these two shift rules to the finite-difference derivative commonly used for
# numerical differentiation. We choose a shift value of :math:`d_x=5\cdot10^{-5}`.


dx = 5e-5
ps_der1 = [parameter_shift(cost_functions[N], N, 1) for N in Ns]
fd_der1 = [(cost_functions[N](dx/2)-cost_functions[N](-dx/2))/(dx) for N in Ns]
ps_der2 = [parameter_shift(cost_functions[N], N, 2) for N in Ns]
fd_der2 = [
    (
        (cost_functions[N](dx)-cost_functions[N](0))/dx
        -(cost_functions[N](0)-cost_functions[N](-dx))/dx
    )/dx
    for N in range(1, 6)
]
print("Number of qubits/RZ gates:         ", *range(1, 6), sep=" "*9)
print(f"First-order parameter-shift rule:  {np.round(np.array(ps_der1), 6)}")
print(f"First-order finite difference:     {np.round(np.array(fd_der1), 6)}")
print(f"Second-order parameter-shift rule: {np.round(np.array(ps_der2), 6)}")
print(f"Second-order finite difference:    {np.round(np.array(fd_der2), 6)}")


###############################################################################
#
# Automatically differentiated reconstructions
# --------------------------------------------
#
# Above we used the explicit parameter-shift rule formulas Eqs. (26,27).
# However, we can also implement the *reconstruction* method of the odd (even) part as a function.
# Using PennyLane's automatic differentiation backends, this then enables us to obtain the correct
# odd-order (even-order) derivatives at the point of interest.
#
# todo: wording in the following:
# If you are interested, here is how to do this, using the modified Dirichlet kernels from App. A2
# in the paper and equidistant shifts.
# For the odd reconstruction this is
#
# .. math ::
#
#   E_\text{odd}(x) &= \sum_{\mu=1}^R E_\text{odd}(x_\mu) \tilde{D}_\mu(x)\\
#   \tilde{D}_\mu(x) &= \frac{\sin(R (x-x_\mu))}{2R \tan\left(\frac{1}{2} (x-x_\mu)\right)} - \frac{\sin(R (x+x_\mu))}{2R \tan\left(\frac{1}{2} (x+x_\mu)\right)},
#
# whereas the even part is
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
# We will now implement these equations, using the reformulation
#
# .. math ::
#
#   \frac{\sin(X)}{\tan(Y)}=\frac{X}{Y}\frac{\operatorname{sinc}(X)}{\operatorname{sinc}(Y)}\cos(Y)
#
# for the kernels.


shifts_odd = lambda R: [(2*mu-1)*np.pi/(2*R) for mu in range(1, R+1)]
# Odd linear combination of Dirichlet kernels
D_odd = lambda x, R: np.array([
    (
        sinc(R*(x-shift))/sinc(0.5*(x-shift))*np.cos(0.5*(x-shift))
        -sinc(R*(x+shift))/sinc(0.5*(x+shift))*np.cos(0.5*(x+shift))
    )
    for shift in shifts_odd(R)
])
# Reconstruction of E_odd
def odd_reconstruction_equ(fun, R):
    evaluations = np.array([(fun(shift) - fun(-shift))/2 for shift in shifts_odd(R)])
    return lambda x: np.dot(evaluations, D_odd(x, R))

shifts_even = lambda R: [mu*np.pi/R for mu in range(1, R)]
# Even linear combination of Dirichlet kernels
D_even = lambda x, R: np.array([
    (
        sinc(R*(x-shift))/sinc(0.5*(x-shift))*np.cos(0.5*(x-shift))
        +sinc(R*(x+shift))/sinc(0.5*(x+shift))*np.cos(0.5*(x+shift))
    )
    for shift in shifts_even(R)
])
# Special cases of even kernels
D0 = lambda x, R: sinc(R*x)/(sinc(x/2))*np.cos(x/2)
Dpi = lambda x, R: sinc(R*(x-np.pi))/sinc((x-np.pi)/2)*np.cos((x-np.pi)/2)
# Reconstruction E_even
def even_reconstruction_equ(fun, R):
    _evaluations = np.array([(fun(shift)+fun(-shift))/2 for shift in shifts_even(R)])
    evaluations = np.array([fun(0), *_evaluations, fun(np.pi)])
    kernels = lambda x: np.array([D0(x, R), *D_even(x, R), Dpi(x,R)])
    return lambda x: np.dot(evaluations, kernels(x))


###############################################################################
# Let's now look at these even and odd reconstructions and how they indeed combine into the
# full function (we will use the ``compare_functions`` utility from above for the latter).

odd_reconstructions = [copy.deepcopy(odd_reconstruction_equ(orig, N)) for orig, N in zip(cost_functions, Ns)]
even_reconstructions = [copy.deepcopy(even_reconstruction_equ(orig, N)) for orig, N in zip(cost_functions, Ns)]
summed_reconstructions = [lambda x: odd_recon(x)+even_recon(x) for odd_recon, even_recon in zip(odd_reconstructions, even_reconstructions)]
axs = compare_functions(cost_functions, summed_reconstructions, Ns)

for i, (odd_recon, even_recon) in enumerate(zip(odd_reconstructions, even_reconstructions)):
    E_odd = np.array([odd_recon(x) for x in X])
    E_even = np.array([even_recon(x) for x in X])
    axs[0,N-1].plot(X, E_odd, color='xkcd:brick red')
    axs[0,N-1].plot(X, E_even, color='xkcd:green')
axs[1,0].set_ylabel('$E-(E_{odd}+E_{even})$');


###############################################################################
# Great! The even and odd part indeed combine into the correct function again. But what did we
# gain? Nothing, actually, for the full reconstruction! Quite the opposite, we spent :math:`2R`
# evaluations of :math:`E` on each part, that is :math:`4R` evaluations overall to obtain a
# description of the full function :math:`E`, instead of the :math:`2R+1` evaluations from the
# first approach.
# However, sometimes we might just be interested in the odd or even part of :math:`E` alone,
# for example to compute odd- or even-order derivatives at a chosen point (with respect to which
# the two parts then have to be odd/even).
# Using autodifferentiation as mentioned above, inparticular JAX, we can compute higher-order
# derivatives without precomputing the corresponding shift rules by hand:


from jax import grad
# An iterator, computing the ``order``th derivative of a function ``f`` via JAX
grad_gen = lambda f, order: grad_gen(grad(f), order-1) if order>0 else f

# Compute the first, second, and fifth derivative
for order, name in zip([1, 2, 5], ["First", "Second", "5th"]):
    recons = odd_reconstructions if order%2 else even_reconstructions
    recon_name = "odd " if order%2 else "even"
    cost_grads = [grad_gen(cost_functions, order)(0.) for N in Ns]
    recon_grads = [grad_gen(recon, order)(0.) for recon in recons]
    all_equal = (
        "All entries match" if np.allclose(cost_grads, recon_grads) else "Some entries differ!"
    )
    print(f"{name} derivatives via autograd: {all_equal}")
    print("From the cost functions:       ", np.round(np.array(cost_grads), 5))
    print(f"From the {recon_name} reconstructions: ", np.round(np.array(recon_grads), 5), '\n')


###############################################################################
# And this is all we want to show here about univariate function reconstructions and generalized
# parameter shift rules.
# Note that the techniques above can partially be extended to frequencies that are not
# integer-valued, but many closed form expressions are no longer valid.
# For the reconstruction, the approach via Dirichlet kernels does no longer work in the general
# case and instead the system of equation given in Eq. (16) has to be solved, but with generalized
# frequencies :math:`\{\Omega_\ell\}` instead of :math:`\{\ell\}`.
#
# todo: referrals?
#
#
#Gen Parshift
#Rotosolve (https://arxiv.org/abs/1905.09692)
#Algebraic?
#Sasha?
#(https://arxiv.org/abs/1812.06323))
#Ns vs Ns in optimizer (brute)
#
#links: brute, shgo, RotosolveOptimizer
