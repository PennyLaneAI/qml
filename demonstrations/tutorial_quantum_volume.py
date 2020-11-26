r""".. _quantum_volume:

Quantum volume
==============

.. meta::
    :property="og:description": Learn about quantum volume, and how to 
        compute it.
    :property="og:image": https://pennylane.ai/qml/_images/bloch.png

Twice per year, a project called the TOP500 [#top500]_ releases a list of the
500 most powerful supercomputing systems in the world. However, there is a large
amount of variation in how supercomputers are built. `Some
<https://en.wikipedia.org/wiki/Fugaku_(supercomputer)>`_ use 48-core processors,
while `others <https://en.wikipedia.org/wiki/Sunway_TaihuLight>`_ use processors
with up to 260 cores. The speed of processors will differ, and they may be
connected in different ways. We can't rank them by simply counting the number of
processors. In order to make a fair comparison, we need benchmarking standards
that give us a holistic view of their performance. To that end, the TOP500
rankings are based on something called the LINPACK benchmark [#linpack]_. The
task of the supercomputers is to solve a dense system of linear equations, and
the metric of interest is the rate at which they perform floating-point
operations (FLOPS). Today's top machines reach speeds well into the regime of
hundreds of peta FLOPs! While a single number certainly cannot tell the whole
story, it still gives us insight into the quality of the machines, and provides
a standard so we can compare them.

A similar problem is emerging with quantum computers: we
can't judge quantum computers on the number of qubits alone. Present-day devices
have a number of limitations, an important one being the fairly high gate error
rates. Typically the qubits on a chip are not all connected to each other, so it
may not be possible to perform operations on arbitrary pairs of
them. Considering this, can we tell if a machine with 20 noisy qubits is better
than one with 5 very high-quality qubits? Or if a machine with 8 fully-connected
qubits is better than one with 16 qubits of comparable error rate, but arranged in
a square lattice?  Furthermore, how can we make comparisons between different
types of qubits?

.. figure:: ../demonstrations/quantum_volume/qubit_graph_variety.svg
    :align: center
    :width: 50%

    Which of these qubit hardware graphs is the best?

To compare across all these facets, researchers have proposed a metric called
"quantum volume" [#cross]_. Roughly, the quantum volume is a measure of the
effective number of qubits a processor has. It is calculated by determining the
largest number of qubits on which it can reliably run circuits of a prescribed
type. You can think of it loosely as a quantum analogue of the LINPACK
benchmark. Different quantum computers are tasked with solving the same problem,
and the success will be a function of many properties: error rates, qubit
connectivity, even the quality of the software stack. A single
number won't tell us everything about a quantum computer, but it does establish
a framework for comparing them.

After working through this tutorial, you'll be able to define quantum volume,
explain the problem on which it's based, and run the protocol to compute it!

"""


##############################################################################
#
# Designing a benchmark for quantum computers
# -------------------------------------------
#
# There are many different properties of a quantum computer
# that contribute to the successful execution of a computation. Therefore, we
# must be very explicit about what exactly we are benchmarking, and what is our
# measure of success. In general, to set up a benchmark for a quantum computer
# we need to decide on a number of things [#robin]_:
#
#  1. A family of circuits with a well-defined structure and variable size
#  2. A set of rules detailing how the circuits can be compiled
#  3. A measure of success for individual circuits
#  4. A measure of success for the family of circuits
#  5. (Optional) An experimental design specifying how the circuits are to be run
#
# We'll work through this list in order to see how the protocol for computing
# quantum volume fits within this framework.
#
# The circuits
# ~~~~~~~~~~~~
#
# Quantum volume relates
# to the largest *square* circuit that a quantum processor can run reliably. This benchmark 
# uses *random* square circuits with a very particular form:
#
# .. figure:: ../demonstrations/quantum_volume/model_circuit_cross.png
#     :align: center
#     :width: 60%
#
#     ..
#
#     A schematic of the random circuit structure used in the quantum volume protocol.
#     Image source: [#cross]_.
#
# Specifically, the circuits consist of :math:`d` sequential layers acting on
# :math:`d` qubits. Each layer consists of two parts: a random permutation of
# the qubits, followed by Haar-random SU(4) operations performed on neighbouring
# pairs of qubits. (When the number of qubits is odd, the bottom-most qubit is
# idle while the SU(4) operations run on the pairs. However, it will still be
# incorporated by way of the permutations.) These circuits satisfy the criteria
# in item 1 --- they have well-defined structure, and it is clear how they can be
# scaled to different sizes.
#
# As for the compilation rules of item 2, to compute quantum volume we're
# allowed to do essentially anything we'd like to the circuits in order to
# improve them. This includes optimization, hardware-aware considerations such
# as qubit placement and routing, and even resynthesis by finding unitaries that
# are close to the target, but easier to implement on the hardware [#cross]_.
#
# Both the circuit structure and the compilation highlight how quantum volume is
# about more than just the number of qubits. The error rates will affect the
# achievable depth, and the qubit connectivity contributes through the layers of
# permutations because a very well-connected processor will be able to implement
# these in fewer steps than a less-connected one. Even the quality of the
# software and the compiler plays a role here: higher-quality compilers will
# produce circuits that fit better on the target devices, and will thus produce
# higher quality results.
#
# The measures of success
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have our circuits, we have to define the quantities that will
# indicate how well we're able to run them. For that, we need a problem
# to solve. The problem used for computing quantum volume is called the *heavy output
# generation problem*. It has roots in the proposals for demonstrating quantum
# advantage [#aaronson]_. Many such proposals make use of the properties of
# various random quantum circuit families, as the distribution of the
# measurement outcomes may not be easy to sample using classical
# techniques.
#
# A distribution that is theorized to fulfill this property is the distribution
# of *heavy* output bit strings. Heavy bit strings are those whose outcome
# probabilities are above the median of the distribution. For example, suppose
# we run a two-qubit circuit, and find that the measurement probabilities for
# the output states are as follows:

measurement_probs = {"00": 0.558, "01": 0.182, "10": 0.234, "11": 0.026}

##############################################################################
#
# The median of this probability distribution is 0.208. The heavy bit
# strings are '00' and '10', because these are the two probabilities above the
# median. If we were to run this circuit, the probability of obtaining one of
# the heavy outputs is 0.792, and so we expect that we will see one of these two
# most of the time.
#
# Each circuit in a circuit family has its own heavy output probability. If our
# quantum computer is of high quality, then we should expect to see heavy
# outputs quite often across all the circuits. On the other hand, if it's of
# poor quality and everything is totally decohered, we will end up with output
# probabilities that are roughly all the same, as decoherence will reduce the
# probabilities to the uniform distribution.
#
# The heavy output generation problem quantifies this --- for our family of random
# circuits, do we obtain heavy outputs at least 2/3 of the time on average?
# Furthermore, do we obtain this with high confidence?
#
# This is the basis for quantum volume. Looking back at the criteria for our
# benchmarks, for item 3, the measure of success for each circuit is how often
# we obtain heavy outputs when we run the circuit and take a measurement. For
# item 4, the measure of success for the whole family is whether or not the mean
# of these probabilities is greater than 2/3 with high confidence.
#
# On a related note, it is important to determine what heavy output probability
# we should *expect* to see on average. The intuition for how this can be
# calculated is as follows [#aaronson]_, [#cmu]_.
#
# Suppose that our random square circuits scramble things up enough so that
# the effective operation looks like a Haar-random unitary :math:`U`. Since
# in the circuits we are applying :math:`U` to the all-zero ket, the
# measurement outcome probabilities will be the moduli squared of the
# entries in the first column of :math:`U`.
#
# Now if :math:`U` is Haar-random, we can say something about the form of these
# entries. In particular, they are complex numbers for which both the real and
# imaginary parts are normally distributed with mean 0 and variance :math:`1/2^m`,
# where :math:`m` is the number of qubits. When we take the mod square of such
# numbers, we obtain an *exponential* distribution, such that the measurement
# outcome probabilities are distributed like  :math:`Pr(p) \sim 2^m e^{-2^m p}.` 
# (This is also known as the *Porter-Thomas distribution*.)
#
# We can integrate this distribution to find that the median sits at :math:`\ln 2`.
# We can further compute the expectation value of obtaining something above
# the medium by integrating the distribution :math:`pe^{-p}` from :math:`\ln 2`
# to infinity, to obtain :math:`(1 + \ln 2)/2`. This is the expected heavy
# output probability! Numerically it is around 0.85, and in fact we will observe
# this later on.
#
#
# The benchmark
# ~~~~~~~~~~~~~
#
# Now that we have our circuits and our measures of success, we're ready to
# define the quantum volume.
#
#
# .. admonition:: Definition
#     :class: defn
#
#     The quantum volume :math:`V_Q` of an :math:`n`-qubit processor is defined as
#
#     .. math::
#         \log_2(V_Q) = \min (m, d(m))
#
#     where :math:`m` is a number of qubits, and :math:`d(m)` is the largest
#     square circuit for which we can reliably sample heavy outputs with
#     probability greater than 2/3.
#
# As an example, if we have a 20-qubit device and find that we get heavy outputs
# reliably for up to depth-4 circuits on 4 qubits, then the quantum volume is
# :math:`\log_2 V_Q = 4`. Quantum volume is incremental, as shown below --- we
# gradually work our way up to larger circuits, until we find something we can't
# do.  Very loosely, quantum volume is like an effective number of qubits. Even
# if we have those 20 qubits, only groups of up to 4 of them work well enough
# together to sample from distributions that would be considered hard.
#
# .. figure:: ../demonstrations/quantum_volume/qv_square_circuits.svg
#     :align: center
#     :width: 75%
#
#     ..
#
#     This quantum computer has :math:`\log_2 V_Q = 4`, as the 4-qubit square
#     circuits are the largest ones it can run successfully.
#
#
# The maximum achieved quantum volume has been doubling at an increasing rate. In
# late 2020, the most recent achievements have been :math:`\log_2 V_Q = 6` on
# IBM's 27-qubit superconducting device `ibmq_montreal` [#qv64]_, and
# :math:`\log_2 V_Q = 7` on a Honeywell trapped-ion qubit processor
# [#honeywell]_. A device with an expected quantum volume of :math:`\log_2 V_Q
# = 32` has also been announced by IonQ [#ionq]_, though benchmarking results
# have not yet been published.
#
# .. note::
#
#    In many sources, the quantum volume of processors is reported as
#    :math:`V_Q` explicitly, rather than :math:`\log_2 V_Q` as is the
#    convention in this demo. As such, IonQs processor has the potential for a
#    quantum volume of :math:`2^{32} > 4000000`. Here we use the :math:`\log`
#    because it is more straightforward to understand that they have 32
#    high-quality, well-connected qubits than to understand at first glance the
#    meaning of the volume itself.
#


##############################################################################
#
# Computing the quantum volume
# ----------------------------
#
#
# Equipped with our definition of quantum volume, it's time to actually try and
# compute it ourselves. We'll use the `PennyLane-Qiskit
# <https://pennylaneqiskit.readthedocs.io/en/latest/>`_ plugin to determine the
# volume of a noisy toy device.
#
# Loosely, the protocol for quantum volume consists of three steps:
#
# 1. Construct random square circuits of increasing size
#
# 2. Run those circuits on both a simulator, and on a noisy hardware device
#
# 3. Perform a statistical analysis of the results to determine what size
#    circuits the device can run reliably
#
#
# The largest reliable size will become the :math:`m` in the expression for
# quantum volume.
#
#
# Step 1: construct random square circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that the structure of the circuits above is alternating layers of
# permutations, and random SU(4) operations on pairs of qubits.  Let's implement
# the generation of such circuits in PennyLane.
#
###############################################################################
#
# First we write a function that randomly permutes qubits. We'll do this by
# using numpy to generate a permutation, and then apply it with SWAP gates.

import numpy as np

# For reproducibility
np.random.seed(42)
# Object for random number generation from numpy
rng = np.random.default_rng()

import pennylane as qml

def permute_qubits(num_qubits):
    # A random permutation
    perm_order = list(rng.permutation(num_qubits))

    working_order = list(range(num_qubits))

    # We will permute by iterating through the permutation and swapping
    # things back to their proper place.
    for idx_here in range(num_qubits):
        if working_order[idx_here] != perm_order[idx_here]:
            # Where do we need to send the qubit at this location?
            idx_there = working_order.index(perm_order[idx_here])
            qml.SWAP(wires=[idx_here, idx_there])

            # Update the working order to account for the SWAP
            working_order[idx_here], working_order[idx_there] = (
                working_order[idx_there],
                working_order[idx_here],
            )


##############################################################################
#
# Next, we need to apply SU(4) gates to pairs of qubits. PennyLane doesn't have
# built-in functionality to generate these random matrices, however its cousin
# `StrawberryFields <https://strawberryfields.ai/>`_ does! We will use the
# ``random_interferometer`` method, which can generate unitary matrices uniformly
# at random. (This function actually generates elements of U(4), but they are
# essentially equivalent up to a global phase).

from strawberryfields.utils import random_interferometer

def apply_random_su4_layer(num_qubits):
    for qubit_idx in range(0, num_qubits, 2):
        if qubit_idx < num_qubits - 1:
            rand_haar_su4 = random_interferometer(N=4)
            qml.QubitUnitary(rand_haar_su4, wires=[qubit_idx, qubit_idx + 1])


##############################################################################
#
# Next, let's write a layering method to put the two together --- this is just
# for convenience and to highlight the fact that these two methods together
# make up one layer of the circuit depth.
#


def qv_circuit_layer(num_qubits):
    permute_qubits(num_qubits)
    apply_random_su4_layer(num_qubits)


##############################################################################
#
# Let's take a look! We'll set up an ideal device with 5 qubits, and
# generate a circuit with 3 qubits. To help with the visualization, we'll also apply
# an RZ gate to each qubit at the beginning to make sure all 5 qubits are there, and
# in order.

num_qubits = 5
dev_ideal = qml.device("default.qubit", analytic=True, wires=num_qubits)

m = 3  # number of qubits

with qml.tape.QuantumTape() as tape:
    for qubit in range(num_qubits):
        qml.RZ(0, wires=qubit)
    qml.templates.layer(qv_circuit_layer, m, num_qubits=m)

print(tape.draw())

##############################################################################
#
# The first thing to note is that the last two qubits are never used in the
# operations, since the quantum volume circuits are square. Another important
# point is that this circuit with 3 layers actually has depth much greater than
# 3, since each layer has both SWAPs and SU(4) operations that are further
# decomposed into elementary gates when run on the actual processor.
#
# One last thing we'll need before running our circuits is the machinery to
# determine the heavy outputs. This is quite an interesting aspect of the
# protocol --- we're required to solve the problem classically in order to get
# the results, so it will only be possible to calculate quantum volume for
# processors up to a certain point before they become too large.
#


def heavy_output_set(m, probs):
    # Sort the probabilities
    probs_ascending_order = np.argsort(probs)
    sorted_probs = probs[probs_ascending_order]

    median_prob = np.median(sorted_probs)

    # Heavy outputs are the bit strings above the median
    heavy_outputs = [
        format(x, f"#0{m+2}b")[2:] for x in list(probs_ascending_order[2 ** (m - 1) :])
    ]

    # Probability of a heavy output
    prob_heavy_output = np.sum(sorted_probs[np.where(sorted_probs > median_prob)])

    return heavy_outputs, prob_heavy_output


##############################################################################
#
# Just as an example, let's compute the heavy outputs and probability for
# our circuit above.
#

with tape:
    qml.probs(wires=range(m))

output_probs = tape.execute(dev_ideal).reshape(
    2 ** m,
)
heavy_outputs, prob_heavy_output = heavy_output_set(m, output_probs)

print(f"Probability of a heavy output is {prob_heavy_output:.6f}")
print(f"Heavy outputs are {heavy_outputs}")


##############################################################################
#
# Step 2: run the circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# First things first, let's set up a hardware device. We'll make a 5-qubit
# processor with the qubits all in a row, then add a little bit of noise.
# Let's take a look at the arrangement of the qubits on the processor by
# plotting its hardware graph.

import networkx as nx

coupling_map = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]

nx.draw_networkx(
    nx.Graph(coupling_map),
    node_color="cyan",
    labels={x: x for x in range(num_qubits)},
)

##############################################################################
#
# This hardware graph is not fully connected, so the compiler will have to make
# some adjustments when non-connected qubits need to interact.
#
# The next thing we need is a noise model. We'll set up a toy one that assumes
# a little bit of depolarizing error on the single-qubit and two-qubit gates,
# as well as some measurement readout error.
from qiskit.providers.aer import noise

noise_model = noise.NoiseModel()

# Add some noise to single qubit gates
for qubit in range(num_qubits):
    # Slightly depolarize u1 and u3 instructions
    u1_error = np.abs(rng.normal(1e-3, 5e-4))
    noise_model.add_quantum_error(noise.depolarizing_error(u1_error, 1), "u1", [qubit])

    u3_error = np.abs(rng.normal(1e-3, 5e-4))
    noise_model.add_quantum_error(noise.depolarizing_error(u3_error, 1), "u3", [qubit])

    # For variety, add probability of a bit flip error on u2 gates
    prob_bit_flip = np.abs(rng.normal(0.05, 1e-3))
    noise_model.add_quantum_error(
        noise.pauli_error([("X", prob_bit_flip), ("I", 1 - prob_bit_flip)]), "u2", [qubit]
    )

# Add some error to the CNOT gates between 0 and the other qubits; roughly 1e-2
for qubit in range(1, num_qubits):
    cnot_error = np.abs(rng.normal(1e-2, 5e-3))
    noise_model.add_quantum_error(noise.depolarizing_error(cnot_error, 2), "cx", [0, qubit])

# Finally, add a touch of measurement readout error; a couple percent for each
for qubit in range(num_qubits):
    meas_error_1if0 = np.abs(rng.normal(2.5e-2, 1e-2))
    meas_error_0if1 = np.abs(rng.normal(2.5e-2, 1e-2))
    readout_error = noise.ReadoutError(
        [[1 - meas_error_1if0, meas_error_1if0], [meas_error_0if1, 1 - meas_error_0if1]]
    )
    noise_model.add_readout_error(readout_error, [qubit])

##############################################################################
#
# Now it's time to create our device. Since this is just a toy model, we'll
# specify 1000 shots. As a final point, since we are allowed to do as much
# optimization as we like, we'll also put the transpiler to work. We'll specify
# some high-quality qubit placement and routing techniques [#sabre]_ in order
# to fit the circuits on the hardware graph in the best way possible.

dev_noisy = qml.device("qiskit.aer", wires=num_qubits, shots=1000, noise_model=noise_model)

dev_noisy.set_transpile_args(
    **{
        "optimization_level": 3,
        "coupling_map": coupling_map,
        "layout_method": "sabre",
        "routing_method": "sabre",
    }
)


##############################################################################
#
# It's time to run the protocol. We'll start with the smallest circuits on 2
# qubits, and make our way up to 5. At each :math:`m`, we'll look at 200 randomly
# generated circuits.
#

min_m = 2
max_m = 5
num_ms = (max_m - min_m) + 1

num_trials = 200

# To store the results
probs_ideal = np.zeros((num_ms, num_trials))
probs_noisy = np.zeros((num_ms, num_trials))

for m in range(min_m, max_m + 1):
    for trial in range(num_trials):

        # Simulate the circuit analytically
        with qml.tape.QuantumTape() as tape:
            qml.templates.layer(qv_circuit_layer, m, num_qubits=m)
            qml.probs(wires=range(m))

        output_probs = tape.execute(dev_ideal).reshape(
            2 ** m,
        )
        heavy_outputs, prob_heavy_output = heavy_output_set(m, output_probs)

        # Execute circuit on the noisy device
        tape.execute(dev_noisy)
        counts = dev_noisy._current_job.result().get_counts()
        reordered_counts = {x[::-1]: counts[x] for x in counts.keys()}

        # Need to flip ordering of the bit strings to match PennyLane
        device_heavy_outputs = np.sum(
            [
                reordered_counts[x] if x[:m] in heavy_outputs else 0
                for x in reordered_counts.keys()
            ]
        )
        fraction_device_heavy_output = device_heavy_outputs / dev_noisy.shots

        probs_ideal[m - min_m, trial] = prob_heavy_output
        probs_noisy[m - min_m, trial] = fraction_device_heavy_output

##############################################################################
#
# Step 3: perform a statistical analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Having run our experiments, we can now get to the heart of the quantum volume
# protocol: what *is* the largest square circuit that our processor can run?
#
# Let's first check out the means, and see how much higher they are than 2/3.


probs_mean_ideal = np.mean(probs_ideal, axis=1)
probs_mean_noisy = np.mean(probs_noisy, axis=1)

print(f"Ideal mean probabilities:")
for idx, prob in enumerate(probs_mean_ideal):
    print(f"m = {idx + min_m}: {prob:.6f} {'above' if prob > 2/3 else 'below'} threshold.")
    
print(f"\nDevice mean probabilities:")
for idx, prob in enumerate(probs_mean_noisy):
    print(f"m = {idx + min_m}: {prob:.6f} {'above' if prob > 2/3 else 'below'} threshold.")

##############################################################################
#
# We see that the ideal probabilities are well over 2/3. In fact, we're quite
# close to the expected value of :math:`(1 + \ln 2)/2 \approx 0.85`.  For the
# device probabilities, however, we see that already by the 4-qubit case we're
# below the threshold. This means that the highest volume this processor can
# have is :math:`\log_2 V_Q = 3`. But isn't enough that just the mean of the
# heavy output probabilities is greater than 2/3. Since we're dealing with
# randomness, we also want to be confident that these results were not just a
# fluke! To be confident, we also want to be above 2/3 within 2 standard
# deviations of the mean. This is referred to as a 97.5% confidence interval
# (since roughly 97.5% of a normal distribution sits within :math:`2\sigma` of
# the mean.)
#
# At this point, we're going to do some statistical sorcery and make some
# assumptions about our distributions. Whether or not a circuit is successful is
# a binary outcome. When we sample many circuits, it is almost like we are
# sampling from a *binomial distribution*, where the outcome probability is
# equivalent to the heavy output probability. In the limit of a large number of
# samples (in this case 200 circuits), a binomial distribution starts to look pretty
# normal. If we make this approximation, we can compute the standard deviation
# and use it to make our confidence interval. With the normal approximation,
# the standard deviation can be computed as
#
# .. math::
#
#    \sigma = \sqrt{\frac{p_h(1 - p_h)}{N}}
#
# where :math:`p_h` is our heavy output probability, and :math:`N` is the number
# of circuits.
#

stds_ideal = np.sqrt(probs_mean_ideal * (1 - probs_mean_ideal) / num_trials)
stds_noisy = np.sqrt(probs_mean_noisy * (1 - probs_mean_noisy) / num_trials)

##############################################################################
#
# Now that we have our standard deviations, we're ready to plot - let's see
# if our means are at least :math:`2\sigma` away from the threshold!
#

import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 6))
ax = ax.ravel()

for m in range(min_m - 2, max_m + 1 - 2):
    ax[m].hist(probs_noisy[m, :])
    ax[m].set_title(f"m = {m + min_m}", fontsize=16)
    ax[m].set_xlabel("Heavy output probability", fontsize=14)
    ax[m].set_ylabel("Occurences", fontsize=14)
    ax[m].axvline(x=2.0 / 3, color="black", label="2/3")
    ax[m].axvline(x=probs_mean_noisy[m], color="red", label="Mean")
    ax[m].axvline(
        x=(probs_mean_noisy[m] - 2 * stds_noisy[m]),
        color="red",
        linestyle="dashed",
        label="2σ",
    )

fig.suptitle("Heavy output distributions for line graph QPU", fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

##############################################################################
#
# In addition to the plot, we should check this numerically.
#

two_sigma_below = probs_mean_noisy - 2 * stds_noisy

for idx, prob in enumerate(two_sigma_below):
    print(f"m = {idx + min_m}: {prob:.6f} {'above' if prob > 2/3 else 'below'} threshold.")

##############################################################################
#
# We see that this is true for for :math:`m=2`, and :math:`m=3`. Thus, we find
# that the quantum volume of this processor is :math:`\log_2 V_Q = 3`.
#
#
# Try playing around with the code yourself: are there any parameters you can
# change to improve the volume with this same noise model? What happens if we
# don't specify a high level of optimization and transpilation? Furthermore, how
# do the results change with a more complex noise model from an actual device?
#
# Concluding thoughts
# -------------------
#
# Quantum volume provides a useful metric for comparing the quality of different
# quantum computers. However, as with any benchmark, it is not without limitations.
# To that end, a more general *volumetric benchmark* framework was proposed, which
# includes not only square circuits, but also rectangular circuits [#robin]_.
#
#
#
# .. _vqe_references:
#
# References
# ----------
#
# .. [#top500]
#
#     `<https://www.top500.org/>`__
#
# .. [#linpack]
#
#    `<https://www.top500.org/project/linpack/>`__
#
# .. [#cross]
#
#    Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D., & Gambetta, J. M.,
#    Validating quantum computers using randomized model circuits, `Physical
#    Review A, 100(3), (2019). <http://dx.doi.org/10.1103/physreva.100.032328>`__
#
# .. [#robin]
#
#    Blume-Kohout, R., & Young, K. C., A volumetric framework for quantum
#    computer benchmarks, `Quantum, 4, 362 (2020).
#    <http://dx.doi.org/10.22331/q-2020-11-15-362>`__
#
# .. [#aaronson]
#
#    Aaronson, S., & Chen, L., Complexity-theoretic foundations of quantum supremacy experiments.
#    `arXiv 1612.05903 quant-ph <https://arxiv.org/abs/1612.05903>`__
#
# .. [#cmu]
#
#    O'Donnell, R. CMU course: Quantum Computation and Quantum Information 2018.
#    `Lecture 25 <https://www.cs.cmu.edu/~odonnell/quantum18/lecture25.pdf>`__
#
# .. [#qv64]
#
#    Jurcevic et al. Demonstration of quantum volume 64 on a superconducting quantum computing system.
#    `arXiv 2008.08571 quant-ph <https://arxiv.org/abs/2008.08571>`__
#
# .. [#honeywell]
#
#    `<https://www.honeywell.com/en-us/newsroom/news/2020/09/achieving-quantum-volume-128-on-the-honeywell-quantum-computer>`__
#
# .. [#ionq]
#
#    `<https://www.prnewswire.com/news-releases/ionq-unveils-worlds-most-powerful-quantum-computer-301143782.html>`__
#
# .. [#sabre]
#
#    Li, G., Ding, Y., & Xie, Y., Tackling the qubit mapping problem for
#    nisq-era quantum devices, `In Proceedings of the Twenty-Fourth
#    International Conference on Architectural Support for Programming Languages
#    and Operating Systems (pp. 1001–1014)
#    (2019). <https://dl.acm.org/doi/10.1145/3297858.3304023>`__ New York, NY,
#    USA: Association for Computing Machinery.
