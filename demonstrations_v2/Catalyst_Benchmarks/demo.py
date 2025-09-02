r"""**Note:** This notebook was tested on Catalyst
`e6ded58 <https://github.com/PennyLaneAI/catalyst/commit/e6ded5866517b2d20ccb729458f949d51b4ac940>`__,
and additionally requires a `small
patch <https://gist.github.com/dime10/a06a5a5add21d4f795ab7bcebf7a460f>`__ in PennyLane.
"""

######################################################################
# Catalyst: Torwards Compilation of FT-Scale Circuits
# ---------------------------------------------------
# 

######################################################################
# A small toy example
# ~~~~~~~~~~~~~~~~~~~
# 
# In this example, we will measure the runtime of simple quantum circuit optimization routines in
# Catalyst’s IR, and compare it to the same routines in PennyLane.
# 
# The circuit will consist of H-H and RX-RX in a loop, which we aim to simplify with adjoint
# cancellation and rotation merging.
# 

import pennylane as qml
from catalyst import qjit, CompileOptions

dev = qml.device("lightning.qubit", wires=2)

# Using a custom compiler pass pipeline will make it easier to measure the runtime of the passes.
compilation_recipe = CompileOptions().get_stages()
peephole_pipeline = ("CIRCUIT_OPT", ["remove-chained-self-inverse{func-name=circuit}",
                                     "merge-rotations{func-name=circuit}"])
compilation_recipe.insert(0, peephole_pipeline)

@qjit(autograph=True, keep_intermediate=True, pipelines=compilation_recipe)
@qml.qnode(dev)
def circuit(theta, loop_size):
    for _ in range(loop_size):
        qml.Hadamard(0)
        qml.Hadamard(0)
        qml.RX(theta, wires=1)
        qml.RX(2*theta, wires=1)
    return qml.probs()

circuit(0.3, 10)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Array([0.04443487, 0.95556513, 0.        , 0.        ], dtype=float64)

######################################################################
# We can look at the original circuit received by Catalyst by inspecting the ``.mlir`` IR attribute:
# 

# Let's filter irrelevant instructions:
criteria = lambda line: any(op in line for op in ["quantum.custom", "scf.for", "arith.add"])
print(*filter(criteria, circuit.mlir.split("\n")), sep="\n")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#          %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %0) -> (!quantum.reg) {
#            %out_qubits = quantum.custom "Hadamard"() %7 : !quantum.bit
#            %out_qubits_0 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
#            %out_qubits_2 = quantum.custom "RX"(%extracted_1) %8 : !quantum.bit
#            %out_qubits_4 = quantum.custom "RX"(%extracted_3) %out_qubits_2 : !quantum.bit

######################################################################
# And looking at the IR after our optimization stage, we can see the effects of: - the cancellation
# pass: no more Hadamards - the merging pass: only one RX gate with an extra addition
# 

from catalyst import debug

new_mlir = debug.get_compilation_stage(circuit, "CIRCUIT_OPT")
print(*filter(criteria, new_mlir.split("\n")), sep="\n")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#          %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %0) -> (!quantum.reg) {
#            %10 = arith.addf %extracted_0, %extracted_1 : f64
#            %out_qubits = quantum.custom "RX"(%10) %8 : !quantum.bit

######################################################################
# Let’s collect some benchmarks!
# ''''''''''''''''''''''''''''''
# 
# We will measure the combined runtime of the two passes in Catalyst using the built-in
# instrumentation, and repeat the measurement for different loop sizes.
# 

import numpy as np

n_samples = 3
loopsizes = np.geomspace(10, 10000, 16, endpoint=True, dtype=int)

open('bench_toy_circuit.yml', 'w').close()  # clear the file if it exists
bench_circuit = circuit.original_function  # use uncompiled circuit to benchmark repeatedly

for n_loops in loopsizes:
    for k in range(n_samples):
        with debug.instrumentation(f"{n_loops}-loops_sample-{k}", filename="bench_toy_circuit.yml", detailed=True):
            qjit(autograph=True, pipelines=compilation_recipe)(bench_circuit)(0.3, n_loops)

######################################################################
# Parsing the collected results from the yaml file:
# 

import yaml
import re

with open('bench_toy_circuit.yml') as f:
    results = yaml.safe_load(f)

cat_data = np.zeros((len(loopsizes), n_samples), dtype=float)
for result in results.values():
    match = re.findall(r'\d+', result['name'])
    n_loops, sample = map(int, match)
    idx = np.argmax(loopsizes == n_loops)
    cat_data[idx][sample] = result['results'][3]['compile']['finegrained'][1]['CIRCUIT_OPT']['walltime']

cat_data[:3]

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    array([[0.251667, 0.271834, 0.248792],
#           [0.253875, 0.25675 , 0.276875],
#           [0.266875, 0.252792, 0.240667]])

######################################################################
# Let’s do the same thing for PennyLane.
# 
# First, we verify that we are measuring what we expect. Note the values and circuits should match
# those from above.
# 

# regular circuit
print(bench_circuit(0.3, 10))
tape = qml.workflow.construct_tape(bench_circuit)(0.3, 1)
print(tape.operations)

# optimized circuit
t1 = qml.transforms.cancel_inverses(tape)[0][0]
t2 = qml.transforms.merge_rotations(t1)[0][0]
print(t2.operations)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    [0.04443487 0.95556513 0.         0.        ]
#    [H(0), H(0), RX(0.3, wires=[1]), RX(0.6, wires=[1])]
#    [RX(np.float64(0.8999999999999999), wires=[1])]

######################################################################
# Now the measurements:
# 

import time

pl_data = np.zeros((len(loopsizes), n_samples), dtype=float)
for idx, n_loops in enumerate(loopsizes):
    for sample in range(n_samples):
        tape = qml.workflow.construct_tape(bench_circuit)(0.3, n_loops)

        start = time.time_ns()
        _ = qml.transforms.merge_rotations(qml.transforms.cancel_inverses(tape)[0][0])
        stop = time.time_ns()

        result = (stop-start) / 1e6  # get time in ms
        pl_data[idx][sample] = result

pl_data[:3]

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    array([[0.429, 0.334, 0.329],
#           [0.469, 0.458, 0.458],
#           [1.106, 0.779, 0.743]])

######################################################################
# Plotting
# ''''''''
# 

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.ticker as mticker

x_data = loopsizes
cat_y_data = np.median(cat_data, axis=1)
cat_y_err = np.std(cat_data, axis=1)
pl_y_data = np.median(pl_data, axis=1)
pl_y_err = np.std(pl_data, axis=1)

fig = plt.figure(figsize=(12, 7.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # 2:1 ratio
ax1 = fig.add_subplot(gs[0])  # First subplot (taller)
ax2 = fig.add_subplot(gs[1])  # Second subplot (shorter)
yellow = plt.cm.tab20(2)
blue = plt.cm.tab20(1)

ax1.set_title("Optimization Runtime on Quantum vs Hybrid IR", fontsize=16)

ax1.errorbar(x_data, cat_y_data, yerr=cat_y_err, marker="o", label="Catalyst", c=yellow, zorder=2)
ax1.errorbar(x_data, pl_y_data, yerr=pl_y_err, marker="s", label="PennyLane", c=blue, ls="-", zorder=2)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("Optimization Time [ms]", fontsize=14)
ax1.set_ylim(1.5e-1, 2e4)
ax1.legend(loc="upper right", fontsize=12, frameon=False)
ax1.tick_params(labelbottom=False, labelleft=True)

text_box = TextArea("$\mathbf{Example~Optimization}$", textprops=dict(color="black", fontsize=14))
ab = AnnotationBbox(text_box, (20, 7000), frameon=False, zorder=1)
ax1.add_artist(ab)
img = mpimg.imread("auto_peephole_comp_horizontal.png")
imagebox = OffsetImage(img, zoom=0.38)
ab = AnnotationBbox(imagebox, (70, 300), frameon=False, zorder=0)
ax1.add_artist(ab)

ax2.plot(loopsizes, pl_y_data/cat_y_data, c=yellow, marker="")
ax2.set_xlabel("Circuit Depth (# iterations)", fontsize=14)
ax2.set_ylabel("Speedup", fontsize=14)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.set_yticks([1, 10, 100, 1000])
ax2.set_yticklabels(["1x", "10x", "100x", "1000x"])
ax2.grid(axis="y", zorder=0)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.show()

######################################################################
#
# .. figure:: ../_static/demonstration_assets/Catalyst_Benchmarks/Catalyst_Benchmarks_c_17_1.png
#    :align: center
#    :width: 80%

######################################################################
# As expected the optimization time remains constant with Catalyst, while it steadily increases in
# PennyLane with the size of the loop in the circuit. Since this is an artificial example, the speedup
# can be made arbitrarily large by increasing the loop count, but it should illustrate the point
# nicely that generating a purely classical circuit before optimization can be costly.
# 
# The next example will demonstrate how costly exactly this can become for a real application.
# 

######################################################################
# Shor’s Algorithm for integer factorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# +---------------------------------------------------------------------------------------------------+
# | :exclamation: The code in this section belongs to Olivia Di Mateo and cannot be shared without    |
# | permission.                                                                                       |
# +===================================================================================================+
# +---------------------------------------------------------------------------------------------------+
# 
# In this section we’ll look at a classic Fault-Tolerant (FT)-era algorithm to test the limits of
# current frameworks. Shor’s algorithm was one of the earliest proposed algorithms with
# super-polynomial speedup, and is still used frequently in benchmarks. The quantum resources required
# to run this algorithm are quite large, and consequently the compilation task can itself be
# bottleneck for larger inputs, and is thus worth benchmarking.
# 

######################################################################
# Since this algorithm is rather complex, let’s define some subroutines in advance:
# 

import pennylane as qml
from jax import numpy as jnp

def modular_inverse(a, N):
    """JIT compatible modular multiplicative inverse routine.

    Source: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Modular_integers
    """
    t = jnp.array(0, dtype=jnp.int32)
    newt = jnp.array(1, dtype=jnp.int32)
    r = jnp.array(N, dtype=jnp.int32)
    newr = jnp.array(a, dtype=jnp.int32)

    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr

    if t < 0:
        t = t + N

    return t

def fractional_binary_to_float(sample):
    """Convert an n-bit sample [k1, k2, ..., kn] to a floating point
    value using fractional binary representation,

        k = (k1 / 2) + (k2 / 2 ** 2) + ... + (kn / 2 ** n)

    Args:
        sample (list[int] or array[int]): A list or array of bits, e.g.,
            the sample output of quantum circuit.

    Returns:
        float: The floating point value corresponding computed from the
        fractional binary representation.
    """
    powers_of_two = jnp.arange(len(sample))
    return jnp.sum(sample * powers_of_two) / 2 ** len(sample)

def as_integer_ratio(f):
    """JIT compatible version of the float.as_integer_ratio() function in Python.

    Converts a floating point number to two 64-bit integers such that their quotient
    equals the input to available precision.
    """
    mantissa, exponent = jnp.frexp(f)

    i = 0
    while jnp.logical_and(i < 300, mantissa != jnp.floor(mantissa)):
        mantissa = mantissa * 2.0
        exponent = exponent - 1
        i += 1

    numerator = jnp.asarray(mantissa, dtype=jnp.int64)
    denominator = jnp.asarray(1, dtype=jnp.int64)
    abs_exponent = jnp.abs(exponent)

    if exponent > 0:
        num_to_return, denom_to_return = numerator << abs_exponent, denominator
    else:
        num_to_return, denom_to_return = numerator, denominator << abs_exponent

    return num_to_return, denom_to_return

def phase_to_order(phase, max_denominator):
    """Estimating which integer values divide to produce a float.

    Given some floating-point phase, estimate integers s, r such
    that s / r = phase, where r is no greater than some specified value.

    Uses a rewritten implementation of the Fraction.limit_denominator method from Python
    suitable for JIT compilation.

    Args:
        phase (float): Some fractional value (here, will be the output
            of running QPE).
        max_denominator (int): The largest r to be considered when looking
            for s, r such that s / r = phase.

    Returns:
        int: The estimated value of r.
    """

    numerator, denominator = as_integer_ratio(phase)

    order = 0

    if denominator <= max_denominator:
        order = denominator

    else:
        p0, q0, p1, q1 = 0, 1, 1, 0

        a = numerator // denominator
        q2 = q0 + a * q1

        while q2 < max_denominator:
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            numerator, denominator = denominator, numerator - a * denominator

            a = numerator // denominator
            q2 = q0 + a * q1

        k = (max_denominator - q0) // q1
        bound1 = p0 + k * p1 / q0 + k * q1
        bound2 = p1 / q1

        loop_res = 0

        if jnp.abs(bound2 - phase) <= jnp.abs(bound1 - phase):
            loop_res = q1
        else:
            loop_res = q0 + k * q1

        order = loop_res

    return order

def QFT(wires):
    shifts = jnp.array([2 * jnp.pi * 2**-i for i in range(2, len(wires) + 1)])

    for i in range(len(wires)):
        qml.Hadamard(wires[i])

        for j in range(len(shifts) - i):
            qml.ControlledPhaseShift(shifts[j], wires=[wires[(i + 1) + j], wires[i]])

def fourier_adder_phase_shift(a, wires):
    """Adds phases on a Fourier-transformed basis state for addition.

    Used as a subroutine by other parts of the modular exponentiation circuits.
    This subroutine assumes that the input is QFT|b>, where |b> is a register
    with n = ceil(log2(b)) + 1 bits (the first of which is overflow).

    After this subroutine, applying QFT^{-1} will yield the state |a + b mod 2^n>
    on n + 1 wires.

    Args:
        a (int): An n-bit integer to be added to a register.
        wires (Wires): The set of wires in the register we are adding to. The
            register should have n+1 wires to prevent overflow.
    """
    # Compute the phases
    n_bits = len(wires)
    a_bits = jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n_bits][
        ::-1
    ]
    powers_of_two = jnp.array([1 / (2**k) for k in range(1, n_bits + 1)])
    phases = (
        2
        * np.pi
        * jnp.array(
            [jnp.dot(a_bits[k:], powers_of_two[: n_bits - k]) for k in range(n_bits)]
        )
    )

    for idx in range(len(wires)):
        qml.PhaseShift(phases[idx], wires=wires[idx])


def doubly_controlled_adder(N, a, control_wires, wires, aux_wire):
    """Doubly controlled Fourier adder, Figure 5 of
    https://arxiv.org/abs/quant-ph/0205095.

    Args:
        N (int): The modulus (number we are trying to factor).
        a (int): An n-bit integer to be added to a register.
        control_wires (Wires): Two wires that this operation is being controlled on.
        wires (Wires): The set of wires in the register we are adding to. The
            register should have n+1 wires to prevent overflow, prepared in some
            basis state QFT|b>.
        aux_wire (Wires): A single wire, used as an auxiliary bit.
    """
    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)

    qml.adjoint(fourier_adder_phase_shift)(N, wires)

    qml.adjoint(QFT)(wires)
    qml.CNOT(wires=[wires[0], aux_wire])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=aux_wire)(N, wires)

    qml.adjoint(qml.ctrl(fourier_adder_phase_shift, control=control_wires))(a, wires)

    qml.adjoint(QFT)(wires)
    qml.PauliX(wires=wires[0])
    qml.CNOT(wires=[wires[0], aux_wire])
    qml.PauliX(wires=wires[0])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)

def controlled_ua(N, a, control_wire, target_wires, aux_wires):
    """Figure 7 of https://arxiv.org/abs/quant-ph/0205095.

    This operation sends |c>|x>|0> to |c>|ax mod N>|0> if c = 1; it is the
    key controlled U_a being applied during phase estimation.

    Note that a must have an inverse modulo N for this function to work.

    Args:
        N (int): The modulus (number we are trying to factor).
        a (int): An n-bit integer to be added to a register.
        control_wire (Wires): The wire this operation is being controlled on.
        target_wires (Wires): The register |x> which should contain the results
            after the subroutine.
        wires (Wires): A set of n + 2 auxiliary wires prepared in |0>
    """
    # Apply controlled multiplication by a mod N; |c>|x>|b>|0> to |c>|x>|(b + ax) mod N>|0>
    QFT(wires=aux_wires[:-1])

    for idx in range(len(target_wires)):
        wire = target_wires[len(target_wires) - idx - 1]
        doubly_controlled_adder(
            N, (a * (2**idx)) % N, [control_wire, wire], aux_wires[:-1], aux_wires[-1]
        )

    qml.adjoint(QFT)(wires=aux_wires[:-1])

    # Controlled SWAP all but the overflow wire; create SWAPS with 2 CNOTs and
    # one Toffoli instead of 3 Toffolis
    for i in range(len(target_wires)):
        t_wire, a_wire = target_wires[i], aux_wires[1 + i]
        qml.CNOT(wires=[a_wire, t_wire])
        qml.Toffoli(wires=[control_wire, t_wire, a_wire])
        qml.CNOT(wires=[a_wire, t_wire])

    # Adjoint of controlled multiplication, but with the modular inverse of a
    mod_inv = modular_inverse(a, N)

    QFT(wires=aux_wires[:-1])

    for idx in range(len(target_wires)):
        wire = target_wires[len(target_wires) - idx - 1]
        qml.adjoint(doubly_controlled_adder)(
            N,
            (mod_inv * (2**idx)) % N,
            [control_wire, wire],
            aux_wires[:-1],
            aux_wires[-1],
        )

    qml.adjoint(QFT)(wires=aux_wires[:-1])

######################################################################
# The main part of the algorithm consists of the quantum phase estimation routine, which can be
# performed efficiently on fewer qubits by reusing the same wire to iteratively estimate bits of the
# phase to desired precision. The estimated phase is then used to compute the factors ``p`` and ``q``
# of ``N`` in a repeat-until-success loop (since the obtained phase is guaranteed to provide the
# correct factors).
# 

import catalyst

# The Catalyst & PL measure function are not unified.
@catalyst.disable_autograph
def measure(w, reset=False, postselect=None):
    if qml.compiler.active():
        return catalyst.measure(w, reset, postselect)
    else:
        return qml.measure(w, reset, postselect)

def shors_algorithm(N, a, n_bits, shots):
    """Execute Shor's algorithm and return a solution.

    Order-finding is essentially QPE with some post-processing. In this function,
    we use a special version of QPE that performs measure-and-reset.

    Args:
        N (int): The number we are trying to factor. Guaranteed to be the product
            of two unique prime numbers.
        a (int): Random integer guess for finding a non-trivial square root.
        n_bits (int): The number of bits in N
        shots (int): The number of shots to take for each candidate value of a

    Returns:
        int, int: If a solution is found, returns p, q such that N = pq. Otherwise
        returns 0, 0.
    """
    # We need 3 registers with 2n + 3 qubits total.
    # - one wire at the top which we measure and reset for QPE (0)
    # - the target wires, upon which ctrl'ed mod. expo. is applied (1, ..., n+1)
    # - a set of aux wires for mod. expo. (n+1, ..., 2n + 3)
    estimation_wire = 0
    target_wires = jnp.arange(n_bits) + 1
    aux_wires = jnp.arange(n_bits + 2) + n_bits + 1

    dev = qml.device("lightning.qubit", wires=2 * n_bits + 3, shots=1)

    @qml.qnode(dev)
    def run_qpe(a):
        # Perform QPE using a single estimation qubit. After controlling the modular
        # exponentation, the qubit is rotated based on previous measurement results,
        # measured, and reset. The measurement outcomes are used to estimate the phase.
        meas_results = jnp.zeros((n_bits,), dtype=jnp.int32)
        cumulative_phase = jnp.array(0.0)
        phase_divisors = 2.0 ** jnp.arange(n_bits + 1, 1, -1)

        qml.PauliX(wires=target_wires[-1])

        for i in range(n_bits):
            exponent = a ** (2 ** ((n_bits - 1) - i))

            qml.Hadamard(wires=estimation_wire)
            controlled_ua(N, exponent, estimation_wire, target_wires, aux_wires)
            qml.PhaseShift(cumulative_phase, wires=estimation_wire)
            meas_results[i] = measure(estimation_wire, reset=True)

            # Compute the corrective phase in the rotation prior to measurement
            cumulative_phase = (
                -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, i + 1))
            )

        return meas_results

    shot_idx = 0

    p, q = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)

    while p * q != N and shot_idx < shots:
        sample = run_qpe(a)
        phase = fractional_binary_to_float(sample)
        guess_r = phase_to_order(phase, N)

        # If the guess order is even, we may have a non-trivial square root.
        # If so, try to compute p and q.
        if guess_r % 2 == 0:
            guess_square_root = (a ** (guess_r // 2)) % N

            if guess_square_root != 1 and guess_square_root != N - 1:
                p = jnp.gcd(guess_square_root - 1, N).astype(jnp.int32)

                if p != 1:
                    q = N // p
                else:
                    q = jnp.gcd(guess_square_root + 1, N).astype(jnp.int32)

                    if q != 1:
                        p = N // q
        shot_idx += 1

    return p, q, shot_idx

######################################################################
# Running it to see how it works. In practice, you’ll want to pick random guesses for ``a`` until you
# find one whose GCD with ``N`` is 1. We’ll limit the number of guesses as well as the number of tries
# per guess, just to ensure the function doesn’t run forever.
# 

import numpy as np

N = 15
max_a = 10
max_trials_per_a = 10

compiled_shors = qml.qjit(shors_algorithm, autograph=True, static_argnums=[2, 3])

print(f"Factoring N={N} ...")
num_a_tried = 0
while num_a_tried < max_a:
    a = np.random.randint(2, N-1)

    print(f"  Trying a={a}")

    if np.gcd(a, N) == 1:
        n_bits = int(jnp.ceil(jnp.log2(N)))
        p, q, shot = compiled_shors(N, a, n_bits, 10)

        if p * q == N:
            print(f"  Found p={p} q={q} after {shot} shots")
            break
        else:
            print(f"  No solution found after {shot} shots, trying new 'a'")

    num_a_tried += 1

assert int(p * q) == N
print("Success!")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Factoring N=15 ...
#      Trying a=6
#      Trying a=4
#      No solution found after 10 shots, trying new 'a'
#      Trying a=12
#      Trying a=7
#      Found p=3 q=5 after 6 shots
#    Success!

######################################################################
# Benchmark time!
# '''''''''''''''
# 
# Let’s start by measuring how far we can reasonably compile this circuit with PennyLane (here we’ll
# chose 1 minute to keep the notebook running time down). We’ll isolate just the circuit portion for
# this:
# 

import time
import sympy

# max we can handle without implementing bigint arithmetic is <64 bits
bit_values = np.linspace(4, 54, 26, dtype=int)
N_values = [sympy.nextprime(2**(n_bits-1)) for n_bits in bit_values]

n_samples = 3
n_datapoints = 6  # 1 minute per datapoint cut off

# This circuit is slightly modified from the above to be compatible with PL.
def qpe_circuit(N, a, n_bits):
    measurements = []
    estimation_wire = 0
    target_wires = jnp.arange(n_bits) + 1
    aux_wires = jnp.arange(n_bits + 2) + n_bits + 1

    qml.PauliX(wires=target_wires[-1])

    for i in range(n_bits):
        exponent = a ** (2 ** ((n_bits - 1) - i))

        qml.Hadamard(wires=estimation_wire)

        controlled_ua(N, exponent, estimation_wire, target_wires, aux_wires)

        for meas_idx, meas in enumerate(measurements):
            qml.cond(meas, qml.PhaseShift)(
                -2 * jnp.pi / 2 ** (i + 2 - meas_idx), wires=estimation_wire
            )

        measurements.append(qml.measure(estimation_wire, reset=True))

    return qml.sample(measurements)

pl_data = np.zeros((n_datapoints, n_samples), dtype=float)
for idx, (n_bits, N) in enumerate(zip(bit_values[:n_datapoints], N_values[:n_datapoints])):
    print(f"measuring {n_bits} bits...")

    a = 2  # value not relevant for benchmark
    n_wires = 2*n_bits + 3

    dev = qml.device("lightning.qubit", wires=n_wires, shots=1)

    for k in range(n_samples):
        qnode_function = qml.qnode(dev)(qpe_circuit)

        # We want to measure all the QNode & device pre-processing, without including execution.
        with catalyst.utils.patching.Patcher((qml, "execute", lambda *args, **kwargs: [None])):
            start = time.time_ns()
            qnode_function(N, a, n_bits)
            stop = time.time_ns()

        result = (stop-start) / 1e9  # get time in s
        pl_data[idx][k] = result

pl_data[:3]

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    measuring 4 bits...
#    measuring 6 bits...
#    measuring 8 bits...
#    measuring 10 bits...
#    measuring 12 bits...
#    measuring 14 bits...# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 

######################################################################
# Let’s just make sure we are indeed not counting the execution time, that is we expect these numbers
# to be higher than the ones above:
# 

for n_bits, N in zip(bit_values[:3], N_values[:3]):
    a = 2  # value not relevant for benchmark
    n_wires = 2*n_bits + 3

    dev = qml.device("lightning.qubit", wires=n_wires, shots=1)
    qnode_function = qml.qnode(dev)(qpe_circuit)

    start = time.time_ns()
    qnode_function(N, a, n_bits)
    stop = time.time_ns()

    result = (stop-start) / 1e9  # get time in s
    print(f"execution on {n_bits} bits took {result}s")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    execution on 4 bits took 0.979538s
#    execution on 6 bits took 3.490371s
#    execution on 8 bits took 36.94229s

######################################################################
# Great!
# 
# Now, let’s do the same with Catalyst, this time measuring the compilation time of the entire hybrid
# algorithm.
# 

from pennylane_lightning.lightning_qubit import LightningQubit

# We need to mock out Lightning's statevector initialization that happens during the Python
# device creation, as that includes allocating a memory buffer which will limit the benchmark size.
def mock_state_init(self):
    self.LightningStateVector = lambda *args, **kwargs: None

cat_data = np.zeros((len(bit_values), n_samples), dtype=float)
for idx, (n_bits, N) in enumerate(zip(bit_values, N_values)):
    if n_bits % 4 == 0:
        print(f"measuring {n_bits} bits...")

    for k in range(n_samples):
        a = 2  # value not relevant for benchmark
        max_trials_per_a = 10
        jit_function = qml.qjit(shors_algorithm, autograph=True, static_argnums=[2, 3])

        with catalyst.utils.patching.Patcher(
            (LightningQubit, "_set_lightning_classes", mock_state_init)
        ):
            start = time.time_ns()
            jit_function.jit_compile((N, a, n_bits, max_trials_per_a))
            stop = time.time_ns()

        result = (stop-start) / 1e9  # get time in s
        cat_data[idx][k] = result

cat_data[:3]

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    measuring 4 bits...
#    measuring 8 bits...
#    measuring 12 bits...
#    measuring 16 bits...
#    measuring 20 bits...
#    measuring 24 bits...
#    measuring 28 bits...
#    measuring 32 bits...
#    measuring 36 bits...
#    measuring 40 bits...
#    measuring 44 bits...
#    measuring 48 bits...
#    measuring 52 bits...# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 

######################################################################
# Check Results
# '''''''''''''
# 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

cat_x_data = bit_values
cat_y_data = np.median(cat_data, axis=1)
cat_y_err = np.std(cat_data, axis=1)
pl_x_data = bit_values[:n_datapoints]
pl_y_data = np.median(pl_data, axis=1)
pl_y_err = np.std(pl_data, axis=1)

plt.figure(figsize=(12, 4.8))
yellow = plt.cm.tab20(2)
blue = plt.cm.tab20(1)

plt.title("Shor's Algorithm in PennyLane & Catalyst", fontsize=16)
plt.errorbar(cat_x_data, cat_y_data, yerr=cat_y_err, marker="o", label="Catalyst", c=yellow, zorder=2)
plt.errorbar(pl_x_data, pl_y_data, yerr=pl_y_err, marker="s", label="PennyLane", c=blue, ls="-", zorder=2)

plt.xlabel("# bits in $N$", fontsize=14)
plt.ylabel("Compilation Time [s]", fontsize=14)
plt.legend(loc="upper right", fontsize=14, frameon=False)

plt.tight_layout()
plt.show()

######################################################################
#
# .. figure:: ../_static/demonstration_assets/Catalyst_Benchmarks/Catalyst_Benchmarks_c_33_1.png
#    :align: center
#    :width: 80%

######################################################################
# We can see that the cost of processing and compiling the circuit grows dramatically in non-hybrid
# frameworks like PennyLane. Compare this to the slow linear growth in compilation time demonstrated
# by Catalyst. In theory, the slope should actually be 0, an indicator that we don’t yet realize 100%
# of the potential of hybrid compilation in Catalyst.
# 

######################################################################
# What about FT application scales?
# '''''''''''''''''''''''''''''''''
# 
# A common proposed application for Shor’s algorithm is the breaking of 2048bit RSA keys. To get a
# sense of the resources required for such large keys, let’s extrapolate out the measurements we just
# took out to 1000 bits. We’ll assume linear growth for Catalyst, and degree 4 polynomial growth for
# PennyLane based on the assumption that the processing time is proportional to the total number of
# operations, which based on the looping patterns in the algorithm grows quartically.
# 

import matplotlib.ticker as mticker
from numpy.polynomial import Polynomial as Poly

cat_fit = Poly.fit(cat_x_data, cat_y_data, deg=1)
pl_fit = Poly.fit(pl_x_data, pl_y_data, deg=3)

extrap_x_data = np.geomspace(4, 1000, 100, endpoint=True)
extrap_cat_y_data = cat_fit(extrap_x_data)
extrap_pl_y_data = pl_fit(extrap_x_data)

fig = plt.figure(figsize=(12, 7.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # 2:1 ratio
ax1 = fig.add_subplot(gs[0])  # First subplot (taller)
ax2 = fig.add_subplot(gs[1])  # Second subplot (shorter)
yellow = plt.cm.tab20(2)
blue = plt.cm.tab20(1)

ax1.set_title("Shor's Algorithm at Application Scale", fontsize=16)

ax1.errorbar(cat_x_data, cat_y_data, yerr=cat_y_err, marker="o", label="Catalyst (data)", c=yellow, ls="", zorder=2)
ax1.errorbar(pl_x_data, pl_y_data, yerr=pl_y_err, marker="s", label="PennyLane (data)", c=blue, ls="", zorder=2)

ax1.plot(extrap_x_data, extrap_cat_y_data, marker="", label="Catalyst (extrapolated)", c=yellow, ls="--", zorder=2)
ax1.plot(extrap_x_data, extrap_pl_y_data, marker="", label="PennyLane (extrapolated)", c=blue, ls="--", zorder=2)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("Compilation Time [s]", fontsize=14)
ax1.legend(loc="upper left", fontsize=12, frameon=False)
ax1.tick_params(labelbottom=False, labelleft=True)

ax2.plot(extrap_x_data, extrap_pl_y_data/extrap_cat_y_data, c=yellow, marker="")
ax2.set_xlabel("# bits in $N$", fontsize=14)
ax2.set_ylabel("Speedup", fontsize=14)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.set_yticks([1, 10, 100, 1000, 10000, 100000])
ax2.set_yticklabels(["1x", "10x", "100x", "1,000x", "10,000x", "100,000x"])
ax2.grid(axis="y", zorder=0)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.show()

######################################################################
#
# .. figure:: ../_static/demonstration_assets/Catalyst_Benchmarks/Catalyst_Benchmarks_c_36_1.png
#    :align: center
#    :width: 80%

######################################################################
# These results illustrate why non-hybrid compilation frameworks cannot hope reach useful application
# scales if the compilation time needs to be measured not in minutes or hours, but years!
# 

######################################################################
# Bonus: Accelerated Workflow Execution
# -------------------------------------
# 

######################################################################
# Variational Algorithms
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# For this example we’ll look at VQE in order to compute the ground state energies of molecules.
# Variational algorithms are particularly suited for JIT compilation due the repeated execution of a
# program with different parameters, which makes the upfront cost of compilation worth it.
# 

# Due to threading overheads, this algorithm can have drastically worse performance when run through
# PennyLane+Lightning, depending on the following factors:
# - OS type: on Linux, Lightning will use threading in its adjoint differentiation method
# - CPU cores: the more cores, the higher the threading overhead
# - problem size: in the smallish qubit regime (<20), threading overheads will be more dominant
# To reduce the variation across environments, we set the number of threads to 1.
# %env OMP_NUM_THREADS=1

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    env: OMP_NUM_THREADS=1

import pennylane as qml
import numpy as np

mols_basis_sets = [
    ["H2", "STO-3G"],  #  4 /   15
    ["H4", "STO-3G"],  #  8 /  185
]
datasets = [qml.data.load("qchem", molname=mol, basis=basis_set)[0] for mol, basis_set in mols_basis_sets]
hams = [dataset.hamiltonian for dataset in datasets]
hf_states = [dataset.hf_state for dataset in datasets]
wire_sets = [ham.wires for ham in hams]
electron_numbers = [dataset.molecule.n_electrons for dataset in datasets]


def get_workflow(n_electron, wires, hf_state, ham):
    singles, doubles = qml.qchem.excitations(n_electron, len(wires))

    dev = qml.device("lightning.qubit", wires=wires, batch_obs=True)
    @qml.qnode(dev, diff_method="adjoint")
    def cost(weights):
        qml.templates.AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
        return qml.expval(ham)

    return cost, len(singles) + len(doubles)

def run_workflow(cost_fn, params, n_iter, step_size):
    opt = qml.GradientDescentOptimizer(stepsize=step_size)
    for _ in range(n_iter):
        params, _ = opt.step_and_cost(cost_fn, params)
    return params


cost_fn, param_size = get_workflow(electron_numbers[0], wire_sets[0], hf_states[0], hams[0])

np.random.seed(42)
params = qml.numpy.array(np.random.normal(0, np.pi, param_size))

run_workflow(cost_fn, params, n_iter=10, step_size=0.2)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    tensor([ 1.69043547, -0.46153325,  1.37307821], requires_grad=True)

######################################################################
# For Catalyst we need to modify the differentiation code slightly, since PL optimizers only work with
# the numpy interface.
# 

from catalyst import qjit, grad

@qjit(autograph=True, static_argnums=[0])
def run_workflow_cat(cost_fn, params, n_iter, step_size):

    for _ in range(n_iter):
        h = grad(cost_fn)(params)
        params = params - h * step_size

    return params

np.random.seed(42)
params = np.random.normal(0, np.pi, param_size)

run_workflow_cat(cost_fn, params, n_iter=10, step_size=0.2)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Array([ 1.69043547, -0.46153325,  1.37307821], dtype=float64)

######################################################################
# Now, let’s collect some data for this worfklow in PennyLane and Catalyst, using the same Lightning
# simulator for execution.
# 

import time

iterations = np.linspace(100, 500, 5, endpoint=True, dtype=int)
n_samples = 5

pl_dataset = {mol: np.empty((len(iterations), n_samples), dtype=float) for mol, _ in mols_basis_sets}
for mol, pl_data in enumerate(pl_dataset.values()):
    for idx, n_iter in enumerate(iterations):
        print(f"measuring {mols_basis_sets[mol][0]} at {n_iter} iterations...")
        for sample in range(n_samples):
            cost_fn, param_size = get_workflow(electron_numbers[mol], wire_sets[mol], hf_states[mol], hams[mol])

            np.random.seed(42)
            params = qml.numpy.array(np.random.normal(0, np.pi, param_size))

            start = time.time_ns()
            run_workflow(cost_fn, params, n_iter, step_size=0.2)
            stop = time.time_ns()

            result = (stop-start) / 1e9  # get time in s
            pl_data[idx][sample] = result
pl_data[:3]

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    measuring H2 at 100 iterations...
#    measuring H2 at 200 iterations...
#    measuring H2 at 300 iterations...
#    measuring H2 at 400 iterations...
#    measuring H2 at 500 iterations...
#    measuring H4 at 100 iterations...
#    measuring H4 at 200 iterations...
#    measuring H4 at 300 iterations...
#    measuring H4 at 400 iterations...
#    measuring H4 at 500 iterations...# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 

cat_dataset = {mol: np.empty((len(iterations), n_samples), dtype=float) for mol, _ in mols_basis_sets}
for mol, cat_data in enumerate(cat_dataset.values()):
    for idx, n_iter in enumerate(iterations):
        print(f"measuring {mols_basis_sets[mol][0]} at {n_iter} iterations...")
        for sample in range(n_samples):
            cost_fn, param_size = get_workflow(electron_numbers[mol], wire_sets[mol], hf_states[mol], hams[mol])

            np.random.seed(42)
            params = np.random.normal(0, np.pi, param_size)

            bench_workflow = run_workflow_cat.original_function  # use uncompiled workflow to benchmark repeatedly

            start = time.time_ns()
            qjit(autograph=True, static_argnums=[0])(bench_workflow)(cost_fn, params, n_iter, step_size=0.2)
            stop = time.time_ns()

            result = (stop-start) / 1e9  # get time in s
            cat_data[idx][sample] = result
cat_data[:3]

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    measuring H2 at 100 iterations...
#    measuring H2 at 200 iterations...
#    measuring H2 at 300 iterations...
#    measuring H2 at 400 iterations...
#    measuring H2 at 500 iterations...
#    measuring H4 at 100 iterations...
#    measuring H4 at 200 iterations...
#    measuring H4 at 300 iterations...
#    measuring H4 at 400 iterations...
#    measuring H4 at 500 iterations...# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 

######################################################################
# Plotting
# ''''''''
# 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from catalyst import __version__ as version

x_data = iterations
cat_y_data = {mol: np.median(cat_data, axis=1) for mol, cat_data in cat_dataset.items()}
cat_y_err = {mol: np.std(cat_data, axis=1) for mol, cat_data in cat_dataset.items()}
pl_y_data = {mol: np.median(pl_data, axis=1) for mol, pl_data in pl_dataset.items()}
pl_y_err = {mol: np.std(pl_data, axis=1) for mol, pl_data in pl_dataset.items()}

fig = plt.figure(figsize=(10, 7.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  # 2:1 ratio
ax1 = fig.add_subplot(gs[0])  # First subplot (taller)
ax2 = fig.add_subplot(gs[1])  # Second subplot (shorter)

ax1.set_title("VQE Execution Benchmark", fontsize=16)

color_index = 0
for mol, _ in mols_basis_sets:
    ax1.errorbar(x_data, pl_y_data[mol], pl_y_err[mol], label=f"{mol} (PL)", marker="o", c=plt.cm.tab20(color_index + 1), ls="--")
    ax1.errorbar(x_data, cat_y_data[mol], cat_y_err[mol], label=f"{mol} (Cat)", marker="s", c=plt.cm.tab20(color_index))
    color_index += 2

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_ylabel("Total Runtime [s]", fontsize=14)
ax1.legend(loc="upper left", frameon=False, fontsize=12)

color_index = 0
for mol, _ in mols_basis_sets:
    speedup_factor = pl_y_data[mol] / cat_y_data[mol]
    ax2.plot(x_data, speedup_factor, c=plt.cm.tab10(color_index), marker="", zorder=10)
    color_index += 1

ax2.set_xlabel("# Iterations", fontsize=14)
ax2.set_yticks([1, 2, 3, 4, 5, 6])
ax2.set_yticklabels([l.get_text() + "x" for l in ax2.get_yticklabels()])
ax2.set_ylabel("Speedup", fontsize=14)
ax2.grid(axis="y", zorder=0)

plt.tight_layout()
plt.show()

######################################################################
#
# .. figure:: ../_static/demonstration_assets/Catalyst_Benchmarks/Catalyst_Benchmarks_c_48_1.png
#    :align: center
#    :width: 80%