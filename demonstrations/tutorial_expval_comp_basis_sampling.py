"""
Quantum expectation value estimation by computational basis sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Original paper: https://arxiv.org/pdf/2112.07416.pdf
# --------------
#
# Note: the implementation of the algorithm proposed in the paper requires generating
# certain unitaries as described in ref https://arxiv.org/pdf/2104.10220.pdf,
# section SM.3. State initialization routines.
#
# The `prepare_superposed` PennyLane branch is required
# for using the qml.utils.get_unitary_preparing_superposition function.
# --------------

import pennylane as qml
import numpy as np

from functools import reduce
import itertools

# Determines whether or not to have the run by estimating of computing with
# analytic values
estimate = True
num_exec = []



# Utils
#
# Step 0.
# -------
#
# Define utility
def _get_state_vector_from_bitstring(bitstring):
    """Returns the state vector corresponging to a state in binary.

    E.g., [0,0,0] -> array([1, 0, 0, 0, 0, 0, 0, 0])

    Args:
        bitstring (list): a list of binary inputs

    Returns:
        np.ndarray: the state vector corresponding to the state in binary
    """
    zero = [1, 0]
    one = [0, 1]
    x_state = [zero if a == 0 else one for a in bitstring]
    return reduce(np.kron, x_state)

import pennylane as qml

# Assume we want to compute <psi|O|psi>
#
# Step 1.
# -------
#
# Prepare the state |psi> followed by the measurement in the computational basis |n>
# -------

num_wires = 3
L_f = int(1e+6) # Num shots
main_estimate_dev = qml.device('default.qubit', wires=num_wires, shots=L_f)

def prepare_psi():
    qml.RY(1.234, wires=0)
    qml.RY(np.pi/2, wires=1)
    qml.RY(2.3214, wires=2)
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,0])

# An example QNode
@qml.qnode(main_estimate_dev)
def circuit1():

    # Prepare the state |psi>
    prepare_psi()
    qml.Snapshot()

    # Measurement in the computational basis |n> (n = 0, 1, ..., 2^N-1)
    return qml.sample()

# Repeat the measurement L_f times
res = qml.snapshots(circuit1)()

num_exec.append(main_estimate_dev._num_executions)

# Obtain sequence of outcomes, set of x in the paper
all_samples = res['execution_results']
psi = res[0]

# Step 2.
# -------
# Compute three quantities (referred to as lists or sets here).
#
# 1.
#   a) Pick up the most frequent R elements from all the samples (set of x) and
#   b) Sort them into descending order of frequency leading to a rearranged sequence: set of z
#
# Suppose x_r appears T_r times in (set of X).
#
# Compute the following quantities:
# 2. f_r: estimate for the single-weight factor, where f_r = T_r/L_f for r=1,2,..,R
# 3. c_R: estimate for the normalization factor, where c_R = 1/sqrt(sum f_r for r=1,2,..,R)
from collections import Counter

def get_set_of_x(set_of_x):
    """Get the list of outcomes to keep.

    Args:
        set_of_x (dict): the counts representing the sampling results, all samples

    Returns:
        list: z_r_set, list of outcomes to keep
    """
    # Convert outcome arrays to tuples such that we have hashable keys
    samples_hashable = [tuple(samples.tolist()) for samples in all_samples]
    counter_obj = Counter(samples_hashable)
    counts = dict(counter_obj)

    descending_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    # Note that we use that zipping will stop iterating over the dict elements

    # TODO: this R def is a trivial first try, it will need to be tweaked later
    R = len(descending_counts.items())
    z_r_set = {k:v for _, (k,v) in zip(range(R), descending_counts.items())}
    return z_r_set


# Sanity check --- f_r (assumes that R is trivially big enough, i.e., R>= 2^N-1)
dev = qml.device('default.qubit', wires=num_wires, shots=None)

# An example QNode
@qml.qnode(dev)
def circuit_probs():

    # Prepare the state |psi>
    prepare_psi()
    #qml.Snapshot()

    # Measurement in the computational basis |n> (n = 0, 1, ..., 2^N-1)
    return qml.probs(wires=[0,1,2])


states = itertools.product((0, 1), repeat=num_wires)
probs = circuit_probs()

kv_map = {k: v for k, v in zip(states, probs)}

kv_map = {k: v for k, v in sorted(kv_map.items(), key=lambda item: item[1], reverse=True)}

#------

# 1. z_r_set: List of outcomes to keep
# 2. f_r: estimate for the single-weight factor, where f_r = T_r/L_f for r=1,2,..,R

# TODO: uncomment, commented such that we have estimates
if estimate:
    z_r_set = get_set_of_x(all_samples)
    f_r = np.array(list(z_r_set.values())) / L_f
else:
    z_r_set = kv_map
    f_r = np.array(list(z_r_set.values()))

assert np.allclose(np.array(list(kv_map.values())), f_r, atol=10e-3)

# 3. c_R: estimate for the normalization factor, where c_R = 1/sqrt(sum f_r for r=1,2,..,R)
c_R = 1 / np.sqrt(sum(f_r))

R = len(f_r)

sorted_probs = sorted(circuit_probs(), reverse=True)

assert np.allclose(sorted_probs, f_r, atol=10e-3) # Arbitrary tolerance picked, we're sampling


# Step 3.
# -------
#
# Evaluate relevant transition matrix elements <z_r|O|z_r`> (r,r'=1,2,...,R).
A = np.array([[6+0j, 1-2j],[1+2j, -1]])

obs = qml.Hermitian(A, wires=[0]) @ qml.PauliZ(2) @ qml.Hermitian(A, wires=[1])
mx_observable = qml.matrix(obs)

def transition_mx_elements(z_r_set, mx_observable):
    """Assumes that we have computational basis state vectors."""
    obs_overlap = {}
    for idx1, z_r in enumerate(z_r_set.keys()):

        for idx2, z_r_prime in enumerate(z_r_set.keys()):
            powers_of_two = 1 << np.flip(np.arange(len(z_r)))

            state1 = np.array(z_r)
            i1 = state1 @ powers_of_two

            state2 = np.array(z_r_prime)
            i2 = state2 @ powers_of_two

            # Index into the matrix of the observable
            overlap = mx_observable[i1, i2]

            # Store the overlaps using a key of the indices
            obs_overlap[(idx1, idx2)] = overlap

    return obs_overlap


# Relevant transition matrix elements
obs_overlap = transition_mx_elements(z_r_set, mx_observable)

# Sanity check Step 3.
#
# Note: the original implementation assumes comp basis
#
def evaluate_transition_mx_element(z_r, mx_observable, z_r_prime):
    return z_r @ mx_observable @ z_r_prime

def get_obs_overlap(z_r_set, mx_observable):
    obs_overlap = {}
    for idx1, z_r in enumerate(z_r_set.keys()):

        for idx2, z_r_prime in enumerate(z_r_set.keys()):
            z1 = _get_state_vector_from_bitstring(z_r)
            z2 = _get_state_vector_from_bitstring(z_r_prime)
            res = evaluate_transition_mx_element(z1, mx_observable, z2)

            # Store the overlaps using a key of the indices
            obs_overlap[(idx1, idx2)] = res

    return obs_overlap

obs_overlap2 = get_obs_overlap(z_r_set, mx_observable)

# iterate
for k,v in obs_overlap.items():
    assert obs_overlap2[k] == v

# Step 4.
# -------
# Def.
# -------
#
# 1. For r = 2, ..., R, prepare a quantum state U_z1,zr|psi>,
#    followed by the measurement in the computational basis
# 2. Repeat the measurement L_A times
# 3. Count the occurrence of the outcome "0"

def sample_fancy(z1, z2, r_value, shots):
    tape = qml.utils.get_unitary_preparing_superposition(z1, z2, r_value)

    dev = qml.device('default.qubit', wires=num_wires, shots=shots)

    @qml.qnode(dev)
    def circuit():
        prepare_psi()

        # Apply U_z1,zr
        for op in tape.operations:
            qml.apply(op)
        return qml.sample()

    res = circuit()

    # Get execution count
    num_exec.append(dev._num_executions)
    return res

def zeros_distribution(outcomes, shots):
    """Get the distribution of zero outcomes."""
    num_zeros = 0
    for idx in range(len(outcomes)):
        out = all_samples[idx,:]
        if np.allclose(out, np.zeros(num_wires)):
            num_zeros += 1

    return num_zeros / shots

counter = []

def estimate_fancy_element(z1, z2, r_value, shots):
    """Estimate the element of the matrix denoted by a fancy letter (fancy A or fancy B).

    z1 and z2 determine the matrix element to estimate.
    """
    counter.append(1)
    outcomes = sample_fancy(z1, z2, r_value, shots)
    A_r = zeros_distribution(outcomes, shots)
    return A_r

from functools import partial

def get_fancy_capital_letter(fixed_z, r_value, shots, swap=False):
    """Get the fancy letter A or B by getting the overlap of the fixed z vector and the z_r set.

    Args:
        fixed_z (tuple): tuple containing the binary representation of the
            basis state to fix
        swap (bool): whether or not the fixed z should be swapped to the second place
        shots (int): the number of shots to use for estimation

    Returns:

        list: list of elements in the set denoted by the fancy A or B letter
    """
    res = []
    for idx, z_r in enumerate(z_r_set.keys()):
        if fixed_z == z_r:
            continue

        fancy = partial(estimate_fancy_element, shots=shots) if estimate else analytic_fancy
        #A_r = estimate_fancy_element(fixed_z, z_r, r_value, shots)
        #analytic_A_r = analytic_fancy(fixed_z, z_r, r_value)

        A_r = fancy(fixed_z, z_r, r_value) if not swap else fancy(z_r, fixed_z, r_value)

        # Sanity check
        if estimate:
            analytic_A_r = analytic_fancy(fixed_z, z_r, r_value) if not swap else analytic_fancy(z_r, fixed_z, r_value)

            #TODO: double-check: the diff is atol=10e-1
            assert np.allclose(A_r, analytic_A_r, atol=10e-1)

        res.append(A_r)
    return res

def analytic_fancy(z1, z2, r_value):
    """Analytic calculation of the fancy letters A or B serving sanity check
    purposes."""
    tape = qml.utils.get_unitary_preparing_superposition(z1, z2, r_value)

    dev = qml.device('default.qubit', wires=num_wires, shots=None)

    @qml.qnode(dev)
    def circuit():
        prepare_psi()

        # Apply U_z1,zr
        for op in tape.operations:
            qml.apply(op)
        return qml.expval(qml.Projector([0] * num_wires, wires=range(num_wires)))

    return circuit()

# Step 4. A)
# ----------
# A_r
# ----------
z_1 = list(z_r_set.keys())[0]

L_A=L_B=L_A_swap=L_B_swap = 1000
A_r_store = get_fancy_capital_letter(z_1, 0, L_A)

# Step 4. B)
# ----------
# B_r
# ----------
B_r_store = get_fancy_capital_letter(z_1, 1, L_B)

# Step 4.
# -------
# A_r (swapped to get the first column of the G matrix)
# -------
A_r_swapped_store = get_fancy_capital_letter(z_1, 0, L_A, swap=True)


# -------
# B_r (swapped to get the first column of the G matrix)
# -------
B_r_swapped_store = get_fancy_capital_letter(z_1, 1, L_B_swap, swap=True)

# Step 5.
# -------
# Combine f_r, A_r, B_r to estimate interference factors
#
# Eq. (19)
# -------
f_1 = f_r[0]
f_1_arr = np.full(len(f_r[1:]), f_1)
g_r = np.array(A_r_store) + 1j * np.array(B_r_store) - ((1+1j) / 2) * (f_1_arr + f_r[1:])

g_r_swapped = np.array(A_r_swapped_store) + 1j * np.array(B_r_swapped_store) -\
                (1+1j) / 2 * (f_r[1:] + f_1_arr)

G_r = {}

z_r = list(z_r_set)
# 1. First diagonal element
G_r[(0,0)] = f_r[0]

# 2. First row Eq. 19
for i in range(len(g_r)):
    G_r[(0, i+1)] = g_r[i]

    # Sanity check
    # ------------
    # Substituted analytic values:
    #
    vec1 = _get_state_vector_from_bitstring(z_r[0])
    vec2 = _get_state_vector_from_bitstring(z_r[i+1])
    exp = vec1 @ np.outer(psi.conj(), psi) @ vec2
    G_r[(0, i+1)] = exp

# 3. First column - used Eq. 19 with (r,1) indices
for i in range(len(g_r_swapped)):
    G_r[(i+1, 0)] = g_r_swapped[i]

    # Sanity check
    # ------------
    # Substituted analytic values:
    #
    vec1 = _get_state_vector_from_bitstring(z_r[i+1])
    vec2 = _get_state_vector_from_bitstring(z_r[0])
    exp = vec1 @ np.outer(psi.conj(), psi) @ vec2
    G_r[(i+1, 0)] = exp

# Eq. (20)
# -------
#
# r=2,...,R
range_obj = range(1, len(g_r)+1)
for idx1 in range_obj:
    for idx2 in range_obj:
        if idx1 == idx2:
            # 4. Most diagonal elements
            G_r[(idx1, idx2)] = f_r[idx1]
        else:
            # 5. Most off-diagonal elements as per Eq. 19

            # idx -1 because g_r has r indexed 2..R
            first = g_r[idx1-1]
            second = g_r[idx2-1]
            G_r[(idx1, idx2)] = np.conj(first) * second/f_1

            # Sanity check
            # ------------
            # Substituted analytic values:
            #
            vec1 = _get_state_vector_from_bitstring(z_r[idx1])
            vec2 = _get_state_vector_from_bitstring(z_r[idx2])
            exp = vec1 @ np.outer(psi.conj(), psi) @ vec2
            G_r[(idx1, idx2)] = exp

# Sanity check G_r
for k,v in G_r.items():
    i1, i2 = k

    # <zr|psi><psi|zr'>
    z_r = list(z_r_set)
    vec1 = _get_state_vector_from_bitstring(z_r[i1])
    vec2 = _get_state_vector_from_bitstring(z_r[i2])
    exp = vec1 @ np.outer(psi, psi) @ vec2
    assert np.allclose(v, exp, atol=0.1)

# Step 6.
# -------
#
# Estimate the expectation value by substituting in the previous quantities.

expval = c_R ** 2

exp_sum = 0
for idx1, z_r in enumerate(z_r_set.keys()):

    for idx2, z_r_prime in enumerate(z_r_set.keys()):
        key = (idx1, idx2)
        exp_sum += f_r[idx1] * f_r[idx2] * obs_overlap[key] / G_r[key]

print("Estimated expval: ", exp_sum)

# Coding up Eq. 13 specifically
# Step 6.
# -------

expval = c_R ** 2

exp_sum = 0

for idx1, z_r in enumerate(z_r_set.keys()):
    for idx2, z_r_prime in enumerate(z_r_set.keys()):
        vec1 = _get_state_vector_from_bitstring(z_r)
        vec2 = _get_state_vector_from_bitstring(z_r_prime)

        term1 = np.abs(vec1 @ psi) **2
        term2 = np.abs(vec2 @ psi) **2

        # <zr|psi><psi|zr'>
        denom = vec1 @ psi.conj() * psi @ vec2

        exp_sum += term1 * term2 * (vec1 @ mx_observable @ vec2) / denom

#print("Eq. 19. using mx and vecs: ", exp_sum)
print("<psi|O|psi>: ", psi @ mx_observable @ psi)

# Real expval:
import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=num_wires, shots=None)

A = np.array([[6+0j, 1-2j],[1+2j, -1]])

obs = qml.Hermitian(A, wires=[0]) @ qml.PauliZ(2) @ qml.Hermitian(A, wires=[1])

def prepare_psi():
    qml.RY(1.234, wires=0)
    qml.RY(np.pi/2, wires=1)
    qml.RY(2.3214, wires=2)
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,0])

@qml.qnode(dev)
def circuit2():
    prepare_psi()
    return qml.expval(qml.Hermitian(qml.matrix(obs), wires=[0,1,2]))

print("Naive run (shots=None): ", circuit2())
print("Number of executions: ", sum(num_exec), sum(counter))
