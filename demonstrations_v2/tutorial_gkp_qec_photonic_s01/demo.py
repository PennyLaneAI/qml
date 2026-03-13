r"""GKP-Based Quantum Error Correction in Photonic Systems
======================================================

**Bundled series:** S01-S05

This bundled demo keeps the original storyline across S01 to S05 in one `demo.py`, with the
section narratives preserved in sequence.
"""

######################################################################
# Introduction
# ------------
# 
# Let’s start with the big picture. Photonic hardware is continuous-variable, but most quantum
# algorithms are written in terms of discrete qubits. A GKP-encoded logical qubit is the bridge
# between those worlds: it hides the oscillator details and exposes a clean two-level abstraction to
# software.
# 
# That abstraction makes programming easier, but it doesn’t make noise disappear. As soon as we encode
# information, errors creep in. So we need error correction to keep the logical information stable.
# 
# There are many codes out there—repetition, Shor, Steane, surface codes, and bosonic codes. For
# photonic systems, bosonic codes are a natural fit. This family includes GKP, cat codes, binomial
# codes, and Fock-state encodings.
# 
# Among these, the Gottesman–Kitaev–Preskill (GKP) code plays a central role. GKP encoding stores
# logical qubits in grid-like structures in phase space, allowing small displacement errors—common in
# photonic systems—to be detected and corrected.
# 
# In this demo we stay at the software layer. We’ll follow a simple flow: logical state → error
# syndrome → correction. We won’t simulate optics; we’ll focus on the logical effect.
# 

######################################################################
# Logical Qubit Model in PennyLane
# --------------------------------
# 
# From a software point of view, a GKP logical qubit can be treated as an effective two-level system
# with a density matrix :math:`\rho`. The messy continuous-variable details live underneath, and their
# net effect shows up as an effective logical noise channel:
# 
# .. math::
# 
# 
#    \rho \;\longrightarrow\; \mathcal{E}(\rho).
# 
# Here :math:`\mathcal{E}` is a completely positive, trace-preserving (CPTP) map that represents
# residual logical errors after correction. This is the architecture-level view used in the original
# GKP proposal and later fault-tolerant extensions [1,2].
# 
# In practice, many different logical noise models are possible. PennyLane [5] provides a range of
# quantum channels for this purpose, including ``qml.PhaseDamping``, ``qml.BitFlip``,
# ``qml.PhaseFlip``, ``qml.AmplitudeDamping``, and ``qml.GeneralizedAmplitudeDamping``. Each
# corresponds to a different way logical information can degrade once the system is viewed as an
# effective qubit.
# 
# In this demo, we focus on the depolarizing channel, implemented in PennyLane as
# ``qml.DepolarizingChannel``. At the mathematical level, it acts as
# 
# .. math::
# 
# 
#    \mathcal{E}_{\text{dep}}(\rho)
#    \;=\;
#    (1 - p)\,\rho
#    \;+
#    \;\frac{p}{3}
#    \left(
#    X \rho X
#    +
#    Y \rho Y
#    +
#    Z \rho Z
#    \right),
# 
# where :math:`p \in [0,1]` is the effective logical noise strength and :math:`X`, :math:`Y`,
# :math:`Z` are the Pauli operators. You can read this as: with total probability :math:`p`, a random
# Pauli error is applied; otherwise the state is left alone.
# 
# The appeal of this model is clarity, not physical realism. In a real photonic system, residual
# logical noise after GKP correction arises from finite squeezing, photon loss, measurement
# imprecision, and imperfect decoding [1,4]. Modeling all of that explicitly would obscure the main
# point here.
# 
# Instead, the depolarizing channel gives us a clean, hardware-agnostic way to represent the net
# outcome of imperfect GKP error correction: a single parameter :math:`p` that tells us how much
# logical noise remains once the physical correction procedures have done their work.
# 

######################################################################
# Requirements
# ------------
# 
# This demo uses PennyLane to illustrate logical noise and error correction at the software level.
# Plots are generated with Matplotlib.
# 
# If PennyLane is not already installed, it can be installed with:
# 
# .. code:: bash
# 
#    pip install pennylane matplotlib
# 

######################################################################
# Case 1: Logical coherence under effective noise
# -----------------------------------------------
# 
#    *Thinking question: “What does logical noise do if we don’t correct it well enough?”*
# 
# Before thinking about error correction, it’s useful to see how logical information degrades in the
# presence of noise.
# 
# From a software perspective, one of the simplest indicators of whether a logical qubit is behaving
# well is its coherence. For a qubit prepared in a superposition state, coherence tells us how
# reliably quantum information can be processed and interfered.
# 
# In this example, we prepare a logical qubit in a superposition using a Hadamard gate and then apply
# an effective logical noise channel. We monitor the expectation value ``⟨X⟩``, which serves as a
# proxy for logical coherence. As the strength of the effective noise increases, we expect this
# coherence to decrease.
# 
# The goal here is not to model the physical noise acting on a photonic system, but to observe how
# residual imperfections, after encoding and (imperfect) error correction, appear at the logical level
# seen by quantum software.
# 

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# Use a mixed-state simulator to model logical noise
dev = qml.device("default.mixed", wires=1)


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 11,
            "lines.linewidth": 2.4,
            "lines.markersize": 5,
        }
    )


apply_plot_style()
colors = {
    "raw": "#1b9e77",
    "corrected": "#d95f02",
}


@qml.qnode(dev)
def logical_gkp_coherence(noise_strength):
    '''Logical qubit prepared in a superposition and subjected to effective logical noise.'''
    qml.Hadamard(wires=0)  # logical Clifford operation
    qml.DepolarizingChannel(noise_strength, wires=0)  # effective logical noise
    return qml.expval(qml.PauliX(0))  # logical coherence


print("Logical GKP qubit circuit (effective model):")
print(qml.draw(logical_gkp_coherence)(0.1))


# --- Sweep effective logical noise strength ---
ps = np.linspace(0.0, 0.30, 61)
coherences = np.array([logical_gkp_coherence(p) for p in ps])

# --- Plot logical coherence decay ---
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ps, coherences, color=colors["raw"], marker="o", markevery=6, label="Logical coherence")
ax.set_xlabel("Effective logical noise strength p")
ax.set_ylabel(r"Logical coherence $\langle X \rangle$")
ax.set_title("Case 1: Coherence decay under effective noise")
ax.set_xlim(ps.min(), ps.max())
ax.set_ylim(0.5, 1.02)
ax.legend()
fig.tight_layout()
plt.show()


######################################################################
#    *What are we seeing here in case 1 above?*
# 
# Let’s walk through the results in plain language.
# 
# We prepare a logical qubit in a superposition using a Hadamard gate. In a noiseless world, this
# state is perfectly coherent. Measuring ``⟨X⟩`` gives ``1.0``, which tells us the superposition is
# intact.
# 
# Now we dial up effective logical noise using the depolarizing channel. This noise is not meant to
# describe the detailed physics of photons; it just captures the net effect of imperfections after
# encoding and imperfect correction.
# 
# As the noise strength ``p`` increases, the trend is clear:
# 
# - When ``p = 0.00``, ``⟨X⟩ = 1.000`` -> the logical qubit is perfectly coherent.
# - As ``p`` increases, ``⟨X⟩`` gradually decreases.
# - By ``p = 0.30``, ``⟨X⟩`` is about ``0.60``.
# 
# This is the simplest picture of logical noise: the circuit is still trivial, but the coherence
# steadily fades.
# 
#    *How do we correct this?*
# 
# In Case 2, we model how GKP correction reduces the effective logical noise.
# 

######################################################################
# Case 2: What changes when error correction does its job?
# --------------------------------------------------------
# 
# In Case 1, we deliberately looked at what happens when effective logical noise is left unchecked.
# The takeaway was simple: as logical noise increases, coherence steadily decays, and the quantum
# state becomes less useful for computation.
# 
# Now let’s flip the question:
# 
#    *What if error correction successfully suppresses logical noise?*
# 
# At the software level, this does not mean that noise disappears completely. Instead, it means that
# the effective logical noise strength is reduced. The circuit stays the same; the noise parameter
# changes.
# 

# Effective logical noise ranges
p_raw = np.linspace(0.0, 0.30, 61)  # before correction
alpha = 0.25  # correction efficiency factor
p_corrected = alpha * p_raw  # after correction

# Compute coherences
coh_raw = np.array([logical_gkp_coherence(p) for p in p_raw])
coh_corrected = np.array([logical_gkp_coherence(p) for p in p_corrected])

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(p_raw, coh_raw, "o-", color=colors["raw"], markevery=6, label="Before correction")
ax.plot(p_raw, coh_corrected, "s--", color=colors["corrected"], markevery=6, label="After correction")
ax.fill_between(p_raw, coh_corrected, coh_raw, color=colors["corrected"], alpha=0.12)
ax.text(0.02, 0.52, rf"$\alpha = {alpha:.2f}$", transform=ax.transAxes, color=colors["corrected"])

ax.set_xlabel("Effective logical noise strength p")
ax.set_ylabel(r"Logical coherence $\langle X \rangle$")
ax.set_title("Case 2: Coherence improvement after GKP correction")
ax.set_xlim(p_raw.min(), p_raw.max())
ax.set_ylim(0.5, 1.02)
ax.legend()
fig.tight_layout()
plt.show()


######################################################################
# In Case 2, we model the effect of GKP error correction by reducing the effective logical noise
# strength according to
# 
# .. math::
# 
# 
#    p_{\text{corrected}} = \alpha \, p_{\text{raw}} .
# 
# Here, ``p_raw`` represents the effective logical noise seen by a qubit before error correction,
# while ``p_corrected`` represents the noise that remains after correction. The parameter
# :math:`\alpha \in (0,1)` captures how effective the error-correction process is at suppressing
# logical errors.
# 
# Intuitively, :math:`\alpha` acts as a correction efficiency factor. Values of :math:`\alpha` closer
# to one correspond to weaker correction, where a large fraction of the logical noise survives.
# Smaller values of :math:`\alpha` correspond to stronger correction, where logical errors are more
# effectively suppressed.
# 
# It is important to emphasize that :math:`\alpha` is not derived from hardware physics in this demo.
# In a real photonic system, its value would depend on concrete physical factors such as squeezing
# levels, photon loss rates, measurement precision, and decoding strategies. Here, we deliberately
# treat :math:`\alpha` as a tunable knob that lets us explore how improved error correction would
# appear at the software level, without committing to a specific hardware implementation.
# 
#    *How should we interpret the result in Case 2?*
# 
# The key thing to notice in the figure above is that the logical circuit itself never changes.
# 
# In both cases, ``before and after correction``, we prepare the same logical qubit, apply the same
# Hadamard gate, and measure the same observable ``⟨X⟩``. There are no additional logical gates, no
# explicit correction steps, and no extra measurements introduced at the circuit level. From the
# software’s point of view, everything looks identical.
# 
# What does change is the effective logical noise associated with the qubit. Before correction,
# increasing logical noise leads to a steady decay of coherence. After correction, the same sweep of
# conditions corresponds to a reduced effective logical noise, and the coherence remains significantly
# higher across the entire range.
# 
# **The separation between the two curves is therefore the software-level signature of GKP error
# correction.**
# 
#    *What is actually happening under the hood?*
# 
# GKP error correction operates on the physical photonic degrees of freedom—continuous variables such
# as small displacements in phase space, well below the level of this circuit. Those physical
# processes never appear explicitly in the logical program. They can, however, introduce substantial
# overhead at the hardware and control layers (syndrome extraction, decoding, feedforward, and time).
# 
# Instead, their net effect is captured by a reduction in the logical noise experienced by the qubit.
# In this demo, that reduction is modeled by scaling the effective logical noise parameter through
# :math:`\alpha`. Smaller values of :math:`\alpha` correspond to more effective correction, while
# larger values indicate that more logical noise remains.
# 
# While the numerical value of :math:`\alpha` is hardware-dependent in practice, the qualitative
# outcome is universal: successful GKP correction suppresses logical errors before the qubit is
# exposed to the program.
# 
#    *Why this matters for quantum software*
# 
# From the perspective of an algorithm designer, error correction is not something you manually
# invoke. It is something that improves the quality of the logical qubits you are given.
# 
# This is why high-level frameworks like PennyLane can treat logical qubits uniformly, regardless of
# whether they come from superconducting devices, trapped ions, or photonic GKP encodings. The
# software interacts with the same abstraction; only the effective noise differs.
# 
#    *Lessons drawn from Case 2*
# 
# Effective error correction shows up at the software level as noise suppression, not circuit
# complexity. GKP encoding allows photonic hardware to deliver logical qubits that behave closer to
# ideal qubits, while keeping the continuous-variable physics hidden beneath the abstraction layer.
# 

######################################################################
# Summary: What we did — and what we didn’t
# -----------------------------------------
# 
#    *What we did*
# 
# We treated a GKP-encoded photonic qubit as a logical two-level system with an effective noise
# channel. Using PennyLane’s ``DepolarizingChannel`` on ``default.mixed``, we saw that:
# 
# - logical coherence decays as effective logical noise increases (Case 1).
# - improved error correction appears as a reduction in that effective noise, leading to higher
#   coherence without changing the circuit (Case 2).
# 
#    *What we didn’t do*
# 
# We did not simulate the physical implementation of GKP error correction. In particular, this demo
# does not include:
# 
# - non-Gaussian continuous-variable simulations of GKP states.
# - explicit syndrome extraction or displacement correction.
# - feedforward operations or decoding circuits.
# - hardware-specific noise models such as photon loss or finite squeezing.
# 
# The correction efficiency parameter :math:`\alpha` is treated as a tunable abstraction, not derived
# from first-principles hardware physics.
# 

######################################################################
# Conclusion
# ----------
# 
# If you’re writing quantum software, GKP error correction shows up as better logical qubits, not as
# extra gates in your program. That’s the key takeaway of this demo.
# 
# We kept the circuit fixed and watched how logical noise affects coherence. Then we modeled
# correction as a reduction in that noise. The result is a clear software-level picture of GKP error
# correction: the logical circuit stays the same, while the effective noise gets smaller.
# 

######################################################################
# AI-use disclosure
# -----------------
# 
# ChatGPT model support was used only for language editing and writing clarity checks.
# 
# Experimental design, implementation, tuning, verification, and all technical conclusions are the
# author’s own work and responsibility.
# 
# All notebook content was reviewed by the author before submission.
# 
# Any opinions, findings, conclusions, or recommendations expressed in this demo are those of the
# author(s) and do not necessarily reflect the views of PennyLane.
# 

######################################################################
# Further reading
# ---------------
# 
# For readers who would like to explore these ideas in more depth, the following references provide
# useful background on GKP encoding, bosonic error correction, and photonic quantum computing:
# 
# - [1] **D. Gottesman, A. Kitaev, and J. Preskill**, *Encoding a qubit in an oscillator*,
#   arXiv:quant-ph/0008040 (2000). https://arxiv.org/abs/quant-ph/0008040
# 
# - [2] **N. C. Menicucci**, *Fault-tolerant measurement-based quantum computing with
#   continuous-variable cluster states*, *Physical Review Letters* **112**, 120504 (2014).
#   https://doi.org/10.1103/PhysRevLett.112.120504
# 
# - [3] **M. Mirrahimi et al.**, *Dynamically protected cat-qubits: a new paradigm for universal
#   quantum computation*, *New Journal of Physics* **16**, 045014 (2014).
#   https://doi.org/10.1088/1367-2630/16/4/045014
# 
# - [4] **M. Banić et al.**, *Exact simulation of realistic Gottesman–Kitaev–Preskill cluster states*,
#   *Physical Review A* **112**, 052425 (2025). https://doi.org/10.1103/PhysRevA.112.052425
# 
# - [5] **V. Bergholm et al.**, *PennyLane: Automatic differentiation of hybrid quantum–classical
#   computations*, arXiv:1811.04968 (2018). https://arxiv.org/abs/1811.04968
# 

######################################################################
# Series continuation (S02)
# -------------------------

######################################################################
# Error correction as noise suppression
# -------------------------------------
# 
# In S01 we treated the logical noise strength ``p`` as a direct knob. In this installment we connect
# that knob to a more physical idea: phase-space displacement noise.
# 
# The circuit stays the same. What changes is how we *generate* the effective logical noise strength
# we feed into it.
# 

######################################################################
# From phase-space noise to logical errors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In photonic systems, a common physical noise process is a small displacement in phase space. We can
# write it as
# 
# .. math::
# 
# 
#    D(\epsilon_q, \epsilon_p),
# 
# where :math:`\epsilon_q` and :math:`\epsilon_p` are small shifts in the position and momentum
# quadratures of the oscillator.
# 
# The GKP code protects against these displacements by measuring syndromes and applying corrective
# displacements. If a physical displacement exceeds the correction threshold, a logical error occurs.
# 
# In this demo we do not simulate the full oscillator dynamics. Instead, we map a *displacement scale*
# to an effective logical noise strength and study how correction suppresses that logical noise.
# 

######################################################################
# Simple correction model
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# To keep things simple, we introduce a toy mapping from a displacement scale :math:`\sigma` to a
# logical error rate:
# 
# .. math::
# 
# 
#    p_{\text{raw}}(\sigma) = 1 - e^{-(\sigma / \sigma_0)^2}.
# 
# This is not derived from hardware physics. It is just a smooth, monotonic map from a physical noise
# scale to a logical error probability. The parameter :math:`\sigma_0` sets the scale.
# 
# We then model correction as a suppression of that logical noise:
# 
# .. math::
# 
# 
#    p_{\text{corrected}} = \\alpha \, p_{\text{raw}},
# 
# with :math:`\\alpha \in (0,1)`. Smaller :math:`\\alpha` means stronger correction.
# 

import os

# Ensure a writable Matplotlib cache and a safe backend for notebooks
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("IPYTHONDIR", "/tmp/ipython")

import matplotlib
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# Use a mixed-state simulator to model logical noise
dev = qml.device("default.mixed", wires=1)


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 11,
            "lines.linewidth": 2.4,
            "lines.markersize": 5,
        }
    )


apply_plot_style()
colors = {
    "raw": "#1b9e77",
    "corrected": "#d95f02",
}


@qml.qnode(dev)
def logical_gkp_coherence(noise_strength):
    """Logical qubit prepared in a superposition and subjected to logical noise."""
    qml.Hadamard(wires=0)
    qml.DepolarizingChannel(noise_strength, wires=0)
    return qml.expval(qml.PauliX(0))


# Physical noise scale (toy model)
sigma = np.linspace(0.0, 0.6, 61)
sigma0 = 0.35

# Map displacement scale to logical noise strength
p_raw = 1.0 - np.exp(-(sigma / sigma0) ** 2)

# Apply correction model
alpha = 0.25
p_corrected = alpha * p_raw

# Compute coherence
coh_raw = np.array([logical_gkp_coherence(p) for p in p_raw])
coh_corrected = np.array([logical_gkp_coherence(p) for p in p_corrected])

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(sigma, coh_raw, "o-", color=colors["raw"], markevery=6, label="Before correction")
ax.plot(sigma, coh_corrected, "s--", color=colors["corrected"], markevery=6, label="After correction")
ax.fill_between(sigma, coh_corrected, coh_raw, color=colors["corrected"], alpha=0.12)
ax.text(0.02, 0.52, rf"$\alpha = {alpha:.2f}$", transform=ax.transAxes, color=colors["corrected"])

ax.set_xlabel(r"Displacement scale $\sigma$ (toy model)")
ax.set_ylabel(r"Logical coherence $\langle X \rangle$")
ax.set_title("S02: Error correction as noise suppression")
ax.set_xlim(sigma.min(), sigma.max())
ax.set_ylim(0.5, 1.02)
ax.legend()
fig.tight_layout()
plt.show()


######################################################################
# What to take away
# ~~~~~~~~~~~~~~~~~
# 
# This demo is still software-level, but now the logical noise strength is tied to a physical noise
# scale. We don’t simulate the oscillator itself, yet we can see how stronger displacement noise leads
# to lower coherence, and how correction suppresses that effect.
# 

######################################################################
# Series continuation (S03)
# -------------------------

######################################################################
# Logical noise model exploration
# -------------------------------
# 
# In S01 and S02 we treated logical noise as one clean knob. That is a good starting point, but real
# logical noise is not always symmetric. This demo asks a more specific question: what kind of logical
# noise is acting on the logical qubit?
# 
# We keep the circuit fixed and only swap the noise channel. The circuit is always: prepare \|+⟩,
# apply a noise channel, then measure ⟨X⟩. We sweep the same noise strength p for every channel so the
# curves are comparable.
# 
# Why this setup? Because it isolates the noise model as the only difference. Any change you see in
# the plot is caused by the channel itself, not by a different circuit or a different measurement.
# 
# If you want to explore, edit the ``channels`` list or the ``ps`` range in the code cell and rerun.
# That is the simplest way to make the comparison more or less aggressive.
# 

import os

# Ensure a writable Matplotlib cache and a safe backend for notebooks
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("IPYTHONDIR", "/tmp/ipython")

import matplotlib
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

# Use a mixed-state simulator to model logical noise
dev = qml.device("default.mixed", wires=1)


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 4.5,
        }
    )


apply_plot_style()


@qml.qnode(dev)
def coherence_with_channel(channel, p):
    qml.Hadamard(wires=0)

    if channel == "depolarizing":
        qml.DepolarizingChannel(p, wires=0)
    elif channel == "bit_flip":
        qml.BitFlip(p, wires=0)
    elif channel == "phase_flip":
        qml.PhaseFlip(p, wires=0)
    elif channel == "amplitude_damping":
        qml.AmplitudeDamping(p, wires=0)
    elif channel == "phase_damping":
        qml.PhaseDamping(p, wires=0)
    else:
        raise ValueError(f"Unknown channel: {channel}")

    return qml.expval(qml.PauliX(0))


channels = [
    ("Depolarizing", "depolarizing"),
    ("Bit flip", "bit_flip"),
    ("Phase flip", "phase_flip"),
    ("Amplitude damping", "amplitude_damping"),
    ("Phase damping", "phase_damping"),
]

ps = np.linspace(0.0, 0.30, 61)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
for label, name in channels:
    coh = np.array([coherence_with_channel(name, p) for p in ps])
    ax.plot(ps, coh, label=label)

ax.set_xlabel("Noise strength p")
ax.set_ylabel(r"Logical coherence $\langle X \rangle$")
ax.set_title("S03: Comparing logical noise channels")
ax.set_xlim(ps.min(), ps.max())
ax.set_ylim(0.5, 1.02)
ax.legend(ncol=2)
fig.tight_layout()
plt.show()


######################################################################
# What we’re measuring and why
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We prepare \|+⟩ because it is maximally coherent in the X basis. Measuring ⟨X⟩ directly reports how
# much of that phase coherence is left after noise.
# 
# Phase-type errors (phase flip, phase damping) attack that coherence directly, so their curves drop
# quickly. Bit-flip noise flips \|0⟩ and \|1⟩ but does not immediately erase X-basis coherence, so it
# can look gentler in this specific measurement. Amplitude damping has its own signature because it
# pushes population toward \|0⟩ while also reducing coherence.
# 
# The key point is that the measurement choice matters. The same physical device can look “more” or
# “less” noisy depending on which logical observable you use to probe it.
# 

######################################################################
# What to take away
# ~~~~~~~~~~~~~~~~~
# 
# The curves do not match, and that is the lesson. Different logical error models degrade coherence in
# different ways, even when they are given the same noise strength parameter.
# 
# If you are trying to model a hardware stack at the logical layer, this plot is a reminder to be
# precise about the channel you choose. “Logical noise” is not one thing. It is a family of models,
# and each one predicts a different coherence decay.
# 
# A practical way to use this demo is to treat the curve shapes as fingerprints. If your measured
# coherence drops quickly in the X basis, phase-type noise is a likely culprit. If it decays more
# slowly, bit-flip-like noise may dominate. The goal is not to fit perfectly here, but to build
# intuition about how channel choice changes the story.
# 

######################################################################
# Series continuation (S04)
# -------------------------

######################################################################
# Multi-qubit logical systems
# ---------------------------
# 
# So far we have looked at single logical qubits. Now we move to entanglement, because that is where
# logical noise really shows its teeth.
# 
# We study two simple states: a Bell state (2 qubits) and a GHZ state (3 qubits). In both cases we
# apply the same logical noise to every qubit and then measure correlation observables.
# 
# These observables are near 1 in the ideal state, so their decay is a direct, software-level signal
# that entanglement is being washed out.
# 
# If you want to explore further, change the ``ps`` range or swap the noise channel in the code cells
# and rerun. You will see how basis choice and noise type change the decay.
# 

import os

# Ensure a writable Matplotlib cache and a safe backend for notebooks
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("IPYTHONDIR", "/tmp/ipython")

import matplotlib
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 4.5,
        }
    )


apply_plot_style()


######################################################################
# Bell state correlations
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# For a Bell state, the correlations ⟨X⊗X⟩ and ⟨Z⊗Z⟩ are both strong. That is why we measure them:
# they are simple, high-contrast indicators that the two qubits are still entangled.
# 
# As noise increases, both correlations decay. The rate of decay is the logical signature of how
# quickly entanglement is lost under the chosen noise channel.
# 

dev2 = qml.device("default.mixed", wires=2)


@qml.qnode(dev2)
def bell_correlations(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)

    xx = qml.expval(qml.PauliX(0) @ qml.PauliX(1))
    zz = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    return xx, zz


ps = np.linspace(0.0, 0.30, 61)
xx_vals = []
zz_vals = []
for p in ps:
    xx, zz = bell_correlations(p)
    xx_vals.append(xx)
    zz_vals.append(zz)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ps, xx_vals, label=r"$\langle X \otimes X \rangle$")
ax.plot(ps, zz_vals, label=r"$\langle Z \otimes Z \rangle$")
ax.set_xlabel("Noise strength p")
ax.set_ylabel("Correlation")
ax.set_title("S04: Bell-state correlations under logical noise")
ax.set_xlim(ps.min(), ps.max())
ax.set_ylim(0.5, 1.02)
ax.legend()
fig.tight_layout()
plt.show()


######################################################################
# How to read the Bell plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Because we use depolarizing noise here, the decay is fairly symmetric and both curves fall together.
# If you swap the noise channel in the code cell to phase flip or bit flip, you will see the basis
# dependence show up as different decay rates between ⟨X⊗X⟩ and ⟨Z⊗Z⟩.
# 
# That is a useful reminder that “entanglement loss” can look different depending on how you probe it.
# 

######################################################################
# GHZ state coherence
# ~~~~~~~~~~~~~~~~~~~
# 
# A GHZ state spreads its coherence across all three qubits, so it is even more fragile. We track
# ⟨X⊗X⊗X⟩ as a proxy for global coherence and ⟨Z0 Z1⟩ as a more local check.
# 
# As noise increases, the global term typically drops faster because it depends on every qubit staying
# coherent at once. This is why multi-qubit logical noise models matter so much for algorithm-level
# behavior.
# 

dev3 = qml.device("default.mixed", wires=3)


@qml.qnode(dev3)
def ghz_correlations(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])

    for w in [0, 1, 2]:
        qml.DepolarizingChannel(p, wires=w)

    xxx = qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2))
    zz = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    return xxx, zz


ps = np.linspace(0.0, 0.30, 61)
xxx_vals = []
zz_vals = []
for p in ps:
    xxx, zz = ghz_correlations(p)
    xxx_vals.append(xxx)
    zz_vals.append(zz)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ps, xxx_vals, label=r"$\langle X \otimes X \otimes X \rangle$")
ax.plot(ps, zz_vals, label=r"$\langle Z_0 Z_1 \rangle$")
ax.set_xlabel("Noise strength p")
ax.set_ylabel("Correlation")
ax.set_title("S04: GHZ coherence under logical noise")
ax.set_xlim(ps.min(), ps.max())
ax.set_ylim(0.5, 1.02)
ax.legend()
fig.tight_layout()
plt.show()


######################################################################
# What to take away
# ~~~~~~~~~~~~~~~~~
# 
# Single-qubit noise already reduces coherence, but entangled states amplify that effect. A small
# amount of logical noise on each wire can erase the correlations that make Bell and GHZ states
# useful.
# 
# This is the software-level reason error correction is essential for multi-qubit algorithms. It is
# not just about keeping individual qubits clean. It is about preserving the correlations that
# algorithms depend on.
# 

######################################################################
# Series continuation (S05)
# -------------------------

######################################################################
# Interactive logical noise playground
# ------------------------------------
# 
# This final demo is a sandbox for everything we built in S01 to S04. You choose a noise model, a
# noise strength p, a correction factor α, and the number of qubits. The circuit and measurement
# update when you rerun the cell; with widgets they update live.
# 
# Under the hood the circuit is simple: for 1 qubit we prepare \|+⟩, for 2 qubits we prepare a Bell
# state, and for 3 qubits we prepare a GHZ state. We then apply the chosen logical noise channel to
# every qubit and measure an X-type observable (⟨X⟩, ⟨X⊗X⟩, or ⟨X⊗X⊗X⟩).
# 
# Correction is modeled as noise suppression: raw noise uses p, corrected noise uses α·p. This is not
# a physical GKP simulation, but it is a clean logical proxy for “correction makes the effective noise
# smaller.”
# 
# In this notebook you can type values directly by editing the ``simulate(...)`` call. If
# ``ipywidgets`` is available, you may also see sliders. The desktop app provides the same controls
# with live plots and typed inputs (no sliders).
# 

import os

# Ensure a writable Matplotlib cache and a safe backend for notebooks
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("IPYTHONDIR", "/tmp/ipython")

import matplotlib
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml


def apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 4.5,
        }
    )


apply_plot_style()


def apply_noise(noise_model, p, wires):
    for w in wires:
        if noise_model == "depolarizing":
            qml.DepolarizingChannel(p, wires=w)
        elif noise_model == "bit_flip":
            qml.BitFlip(p, wires=w)
        elif noise_model == "phase_flip":
            qml.PhaseFlip(p, wires=w)
        elif noise_model == "amplitude_damping":
            qml.AmplitudeDamping(p, wires=w)
        elif noise_model == "phase_damping":
            qml.PhaseDamping(p, wires=w)
        else:
            raise ValueError(f"Unknown noise model: {noise_model}")


def make_circuit(n_qubits, noise_model):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(p):
        if n_qubits == 1:
            qml.Hadamard(wires=0)
        elif n_qubits == 2:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
        else:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        apply_noise(noise_model, p, range(n_qubits))

        if n_qubits == 1:
            return qml.expval(qml.PauliX(0))
        elif n_qubits == 2:
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))
        else:
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2))

    return circuit


def simulate(noise_model="depolarizing", p=0.2, alpha=0.25, n_qubits=2):
    circuit = make_circuit(n_qubits, noise_model)
    raw = circuit(p)
    corrected = circuit(alpha * p)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(["Raw", "Corrected"], [raw, corrected], color=["#1b9e77", "#d95f02"])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Coherence")
    ax.set_title(f"Noise model: {noise_model}, qubits: {n_qubits}")
    for i, val in enumerate([raw, corrected]):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center")
    fig.tight_layout()
    plt.show()


# Try to create interactive controls; fall back to a static example if unavailable
try:
    import ipywidgets as widgets
    from ipywidgets import interact

    interact(
        simulate,
        noise_model=widgets.Dropdown(
            options=[
                "depolarizing",
                "bit_flip",
                "phase_flip",
                "amplitude_damping",
                "phase_damping",
            ],
            value="depolarizing",
            description="Noise",
        ),
        p=widgets.FloatSlider(min=0.0, max=0.5, step=0.05, value=0.2, description="p"),
        alpha=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.25, description="alpha"),
        n_qubits=widgets.Dropdown(options=[1, 2, 3], value=2, description="Qubits"),
    )
except Exception:
    simulate()


######################################################################
# What to try
# ~~~~~~~~~~~
# 
# Start with a baseline so you can calibrate your intuition. Set ``noise_model="depolarizing"``,
# ``p=0.2``, ``alpha=0.25``, ``n_qubits=1``. You should see the corrected bar higher than the raw bar.
# 
# Then try these small experiments, one at a time:
# 
# 1. Hold p fixed and reduce alpha. The corrected bar should rise because you are suppressing noise
#    more aggressively.
# 2. Hold alpha fixed and increase p. Both bars should fall, but the corrected bar should fall more
#    slowly.
# 3. Switch to ``phase_flip`` and compare ``n_qubits=1`` vs ``n_qubits=3``. The three-qubit coherence
#    should collapse faster because it depends on all qubits staying coherent.
# 4. Try ``amplitude_damping`` and notice how it tends to pull states toward \|0⟩, which changes the
#    coherence in a different way than pure phase noise.
# 
# If you want a quick sanity check, set ``alpha=1.0``. The corrected bar should match the raw bar
# exactly.
# 

######################################################################
# Download the desktop playground
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Use the latest release builds:
# 
# - macOS:
#   https://github.com/denniswayo/bosonicflow-gkp/releases/latest/download/bosonicflow-gkp-macos.zip
# - Windows:
#   https://github.com/denniswayo/bosonicflow-gkp/releases/latest/download/bosonicflow-gkp-windows.zip
# - Linux:
#   https://github.com/denniswayo/bosonicflow-gkp/releases/latest/download/bosonicflow-gkp-linux.zip
# 
# If you are running locally from source, see the build scripts in ``pyqt_gui/``.
# 
