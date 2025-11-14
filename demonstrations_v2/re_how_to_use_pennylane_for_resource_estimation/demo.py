r"""How to use PennyLane for Resource Estimation
============================================
"""

######################################################################
# Fault tolerant quantum computing is on its way. But are there useful algorithms which are ready for
# it? The development of meaningful applications of quantum computing is an active area of research,
# and one of the major challenges in the process of assessing a potential quantum algorithm is
# determining the amount of resources required to execute it on hardware. An algorithm may still be
# helpful even when it cannot be executed, but is only truly helpful when it can.
# 
# PennyLane is here to make that process easy, with our new resource estimation module:
# `pennylane.estimator <https://docs.pennylane.ai/en/latest/code/qml_estimator.html>`__.
# 
# `pennylane.estimator <https://docs.pennylane.ai/en/latest/code/qml_estimator.html>`__ is meant to: -
# Make reasoning about quantum algorithms *quick* and painless - no complicated inputs, just tell
# ``estimator`` what you know. - Keep you up to *speed* - ``estimator`` leverages the latest results
# from the literature to make sure you’re as efficient as can be. - Get you moving *even faster* - in
# the blink of an eye ``estimator`` provides you with resource estimates, and enables effortless
# customization to enhance your research.
# 

######################################################################
# Let’s import our quantum resource estimator.
# 

import pennylane as qml
import pennylane.estimator as qre

import numpy as np
import matplotlib.pyplot as plt
import time
import math

######################################################################
# We will be using the Kitaev model as an example to explore resource estimation. See `these
# docs <https://docs.pennylane.ai/en/stable/code/api/pennylane.spin.kitaev.html>`__ for more
# information about the Kitaev hamiltonian. The hamiltonian is defined through nearest neighbor
# interactions on a honeycomb shaped lattice as follows:
# 
# :raw-latex:`\begin{align*}
#   \hat{H} = K_X \sum_{\langle i,j \rangle \in X}\sigma_i^x\sigma_j^x +
#   \:\: K_Y \sum_{\langle i,j \rangle \in Y}\sigma_i^y\sigma_j^y +
#   \:\: K_Z \sum_{\langle i,j \rangle \in Z}\sigma_i^z\sigma_j^z
# \end{align*}`
# 
# In this demo we will estimate the quantum resources required to evolve the quantum state of a 100 x
# 100 unit honeycomb lattice of spins (thats 20,000 spins!) under the Kitaev hamiltonian.
# 

# Construct the hamiltonian on a 30 units x 30 units lattice
n_cells = [30, 30]
kx, ky, kz = (0.5, 0.6, 0.7)

t1 = time.time()
spin_ham = qml.spin.kitaev(n_cells, coupling=np.array([kx, ky, kz]))
t2 = time.time()

print(f"Processing time: ~ {round(t2 - t1)} sec")
print("Total number of terms:", len(spin_ham.operands))
print("Total number of qubits:", len(spin_ham.wires))

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Processing time: ~ 5 sec
#    Total number of terms: 2640
#    Total number of qubits: 1800

######################################################################
# | Notice that it took some time to generate this hamiltonian.
# | Resource estimation is important because even generating a full description of the hamiltonian can
#   be quite computationally expensive. In this case it would take about 15 minutes just to generate
#   the dense description of the hamiltonian!
# 

n_lst = [i for i in range(6,32)]
time_lst = []
for n in n_lst:
    n_cells = [n, n]
    kx, ky, kz = (0.5, 0.6, 0.7)
    
    t1 = time.time()
    spin_ham = qml.spin.kitaev(n_cells, coupling=np.array([kx, ky, kz]))
    t2 = time.time()
    time_lst.append(t2-t1)
    if n % 5 == 0:
        print(f"Finished n = {n} in ~ {time_lst[-1]} sec")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Finished n = 10 in ~ 0.0780797004699707 sec
#    Finished n = 15 in ~ 0.48148012161254883 sec
#    Finished n = 20 in ~ 1.0496199131011963 sec
#    Finished n = 25 in ~ 2.62518310546875 sec
#    Finished n = 30 in ~ 5.784091949462891 sec

plt.plot(n_lst, time_lst, ".", label="Measured processing times")

a, b = (0.0000076, 4)
fit_lst = [
    a * n**b for n in (n_lst + [110])
]
projected_time = a * (100**b)

plt.plot(n_lst + [110], fit_lst, "--g", label="Best-fit")
plt.plot([100], [projected_time], "*r", label=f"n={100}, time ~ {round(projected_time / 60)} mins")

plt.xscale("log")
plt.xlabel("Number of unit cells")
plt.yscale("log")
plt.ylabel("Processing time (sec)")

plt.legend()
plt.show()

######################################################################
#
# .. figure:: ../_static/demonstration_assets/re_how_to_use_pennylane_for_resource_estimation/re_how_to_use_pennylane_for_resource_estimation_28ac98fd-aa4c-478c-9142-5340c86c4fdf_1.png
#    :align: center
#    :width: 80%

######################################################################
# Making it Easy
# ~~~~~~~~~~~~~~
# 

######################################################################
# Thankfully we don’t need a detailed description of our hamiltonian to estimate its resources! The
# geometry of the honeycomb lattice and the structure of the hamiltonian allows us to calculate some
# important quantities directly:
# 
# :raw-latex:`\begin{align}
#   n_{q} = 2 n^{2}, \\
#   n_{YY} = n_{ZZ} = n * (n - 1), \\
#   n_{XX} = n^{2}, \\
# \end{align}`
# 
# We can capture the key information of our hamiltonian in a compact representation using the
# ``qre.PauliHamiltonian`` class.
# 

n_cell = 100
n_q = 2 * n_cell**2
n_xx = n_cell**2
n_yy = n_cell*(n_cell-1)
n_zz = n_yy

distribution_of_pauli_words = {
    "XX": n_xx,
    "YY": n_yy,
    "ZZ": n_zz,
}

kitaev_H = qre.PauliHamiltonian(
    num_qubits = n_q,
    pauli_dist = distribution_of_pauli_words,
)

######################################################################
# Now, ``qre.estimate`` provides us with our circuit resources!
# 

order = 2
num_steps = 10

def circuit(hamiltonian):
    qre.UniformStatePrep(num_states = 2**n_q)    # Prepare a uniform superposition over all 2^num_qubit basis states
    qre.TrotterPauli(hamiltonian, num_steps, order)
    return

print(qre.estimate(circuit)(kitaev_H))

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 3.022E+7
#       'T': 2.622E+7,
#       'CNOT': 1.192E+6,
#       'Z': 3.960E+5,
#       'S': 7.920E+5,
#       'Hadamard': 1.612E+6

# We can estimate the resources of individual operators as well! 
resources_without_grouping = qre.estimate(qre.TrotterPauli(kitaev_H, num_steps, order))


# Commuting groups:
commuting_groups = [    # Alternatively we can split our terms into groups
    {"XX": n_xx},       # of commuting terms, this will help reduce the
    {"YY": n_yy},       # cost of Trotterization as we will see:
    {"ZZ": n_zz},
]

kitaev_H_with_grouping = qre.PauliHamiltonian(
    num_qubits = n_q,
    commuting_groups = commuting_groups,
)

resources_with_grouping = qre.estimate(qre.TrotterPauli(kitaev_H_with_grouping, num_steps, order))


# print("Without grouping:", f"\n{resources_without_grouping}\n")  # [Optionally show this?]
# print("With grouping:", f"\n{resources_with_grouping}")

# Just compare T gates: 
print("With grouping:", f"\n T counts: {resources_with_grouping.gate_counts["T"]:.3E}\n")
print("Without grouping:", f"\n T counts: {resources_without_grouping.gate_counts["T"]:.3E}")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    With grouping:
#     T counts: 1.791E+07
#    
#    Without grouping:
#     T counts: 2.622E+07

######################################################################
# Gatesets & Configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The cost of an algorithm is quantified by the number of logical qubits required and the number of
# gates used. Different hardware will natively support different gatesets. The default gateset used by
# ``estimate`` is ``'Toffoli', 'Hadamard', 'X', 'T', 'S', 'Y', 'Z', 'CNOT'``. We can configure the
# gateset to obtain resource estimates at various levels of abstraction
# 

from pennylane.estimator.resources_base import DefaultGateSet
print("Default gateset:\n", DefaultGateSet)

res = qre.estimate(circuit)(kitaev_H_with_grouping)
print(f"\n{res}")


######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Default gateset:
#     frozenset({'Hadamard', 'X', 'S', 'Y', 'T', 'CNOT', 'Toffoli', 'Z'})
#    
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.116E+7
#       'T': 1.791E+7,
#       'CNOT': 8.140E+5,
#       'Z': 3.960E+5,
#       'S': 7.920E+5,
#       'Hadamard': 1.252E+6

# Customize gateset: 

highlevel_gateset = {
    'RX',
    'RY',
    'RZ',
    'Toffoli', 
    'Hadamard', 
    'X', 
    'T',
    'S', 
    'Y', 
    'Z', 
    'CNOT',
}

high_res = qre.estimate(circuit, gate_set=highlevel_gateset)(kitaev_H_with_grouping)
print(f"High-level resources:\n{high_res}\n")

lowlevel_gateset = {
    'T',
    'CNOT',
    'Hadamard', 
}

low_res = qre.estimate(circuit, gate_set=lowlevel_gateset)(kitaev_H_with_grouping)
print(f"Low-level resources:\n{low_res}")


######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    High-level resources:
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 3.661E+6
#       'RZ': 4.070E+5,
#       'CNOT': 8.140E+5,
#       'Z': 3.960E+5,
#       'S': 7.920E+5,
#       'Hadamard': 1.252E+6
#    
#    Low-level resources:
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.314E+7
#       'T': 2.108E+7,
#       'CNOT': 8.140E+5,
#       'Hadamard': 1.252E+6

######################################################################
# When decomposing our algorithms to a gateset, it is often the case that we only have some
# approximate decomposition of our building-block into the target gateset (e.g approximate state
# loading to some precision, or rotation synthesis within some precision of the rotation angle.
# 
# These approximate decompositions are accurate to within some error threshold; tuning this error
# threshold determines the resource cost of the algorithm. We can set and tune these errors using
# ``ResourceConfig``.
# 

custom_rc = qre.ResourceConfig()
# print(custom_rc)  # This print looks pretty ugly :( 

rz_precisions = custom_rc.resource_op_precisions[qre.RZ]
print(rz_precisions)  # Notice that the default precision for RZ is 1e-9

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    {'precision': 1e-09}

custom_rc.set_precision(qre.RZ, 1e-15)  # setting the required precision from 1e-9 --> 1e-15

res = qre.estimate(
    circuit, 
    gate_set=lowlevel_gateset,
    config=custom_rc,
)(kitaev_H_with_grouping)

# print(f"New low-level resources:\n{res}")  # [Optionally show this?]

# Just compare T gates: 
print("Low precision (1e-9):", f"\n T counts: {low_res.gate_counts["T"]:.3E}\n")
print("High precision (1e-15):", f"\n T counts: {res.gate_counts["T"]:.3E}")     # Notice that a more precise estimate requires more T-gates! 

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Low precision (1e-9):
#     T counts: 2.108E+07
#    
#    High precision (1e-15):
#     T counts: 2.108E+07

######################################################################
# Swapping Decompositions
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# There are many ways to decompose a quantum gate into our target gateset. Selecting an alternate
# decomposition is a great way to optimize the cost of your quantum workflow. This can be done easily
# with the ``ResourceConfig`` class.
# 
# Let’s explore decompositions for the ``RZ`` gate:
# 
# Current decomposition for RZ (single qubit rotation synthesis in general) is: `Efficient Synthesis
# of Universal Repeat-Until-Success Circuits (Bocharov, et al) <https://arxiv.org/abs/1404.5320>`__
# 
# Other state of the art methods we could use instead: - `Optimal ancilla-free Cliﬀord+T approximation
# of z-rotations (Ross, Selinger) <https://arxiv.org/pdf/1403.2975>`__ - `Shorter quantum circuits via
# single-qubit gate approximation (Kliuchnikov et al) <https://arxiv.org/abs/2203.10064v2>`__
# 

default_cost_RZ = qre.estimate(qre.RZ(precision=1e-9))  # We can also manually set the precision
print(default_cost_RZ)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    --- Resources: ---
#     Total wires: 1
#       algorithmic wires: 1
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 44
#       'T': 44

# According to paper by Ross & Selinger, we can decompose RZ rotations into T-gates according to: 

def gridsynth_t_cost(error):
    return round(3 * math.log2(1/error))

# According to paper by Kliuchnikov et al, we can decompose RZ rotations into T-gates according to: 

def mixed_fallback_t_cost(error):
    return round(0.56 * math.log2(1/error) + 5.3)


# In order to define a resource decomposition we first need to know what the resource_keys are for 
#  the operator whose decomposition we want to add

print(qre.RZ.resource_keys)  # this tells us all of the REQUIRED arguments our function must take:

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    {'precision'}

# Now we define our resource decomp: 

def gridsynth_decomp(precision):
    t_resource_rep = qre.resource_rep(qre.T) 
    t_counts = gridsynth_t_cost(precision)
    
    t_gate_counts = qre.GateCount(t_resource_rep, t_counts)  # The GateCounts tell us how many of this type of gate is used in the decomposition
    
    return [t_gate_counts]   # We return a list because there could have been other gates that appear in the decomposition


def mixed_fallback_decomp(precision):
    t_resource_rep = qre.resource_rep(qre.T) 
    t_counts = mixed_fallback_t_cost(precision)
    
    t_gate_counts = qre.GateCount(t_resource_rep, t_counts)  # The GateCounts tell us how many of this type of gate is used in the decomposition
    
    return [t_gate_counts]   # We return a list because there could have been other gates that appear in the decomposition

# Finally we set the new decomposition in our Resource Config 

grisynth_rc = qre.ResourceConfig()
grisynth_rc.set_decomp(qre.RZ, gridsynth_decomp)
grisynth_cost_RZ = qre.estimate(qre.RZ(precision=1e-9), config=grisynth_rc)

mixed_fallback_rc = qre.ResourceConfig()
mixed_fallback_rc.set_decomp(qre.RZ, mixed_fallback_decomp)
mixed_fallback_cost_RZ = qre.estimate(qre.RZ(precision=1e-9), config=mixed_fallback_rc)

print("GridSynth decomposition", f"T counts: {grisynth_cost_RZ.gate_counts["T"]}")
print("Default (RUS) decomposition", f"T counts: {default_cost_RZ.gate_counts["T"]}")
print("Mixed Fallback decomposition", f"T counts: {mixed_fallback_cost_RZ.gate_counts["T"]}")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    GridSynth decomposition T counts: 90
#    Default (RUS) decomposition T counts: 44
#    Mixed Fallback decomposition T counts: 22

######################################################################
# Putting it All Together
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# We can combine all of the features we have seen so far to optimize the cost of Trotterized time
# evolution of the Kitaev hamiltonian:
# 

kitaev_hamiltonian = kitaev_H_with_grouping  # use compact hamiltonian with grouping

custom_gateset = {  # Use the low level gateset
    'T',
    'CNOT',
    'Hadamard', 
}

custom_config = qre.ResourceConfig()
custom_config.set_precision(qre.RZ, precision=1e-12)     # set higher precision 1e-9 --> 1e-12
custom_config.set_decomp(qre.RZ, mixed_fallback_decomp)  # set alternate decomposition 

resources = qre.estimate(circuit, gate_set = custom_gateset, config = custom_config)(kitaev_hamiltonian)
print(resources)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 1.419E+7
#       'T': 1.212E+7,
#       'CNOT': 8.140E+5,
#       'Hadamard': 1.252E+6

######################################################################
# Mapping your PennyLane circuits to ``estimator``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# If you’ve already written your workflow for execution, we can call estimate on it directly. No need
# to write it again!
# 

n_cell = 30
n_cells = [n_cell, n_cell]
kx, ky, kz = (0.5, 0.6, 0.7)

t1 = time.time()
flat_hamiltonian = qml.spin.kitaev(n_cells, coupling=np.array([kx, ky, kz]))
flat_hamiltonian.compute_grouping()  # compute the qubitize commuting groups! 

groups = []
for group_indices in flat_hamiltonian.grouping_indices:
    grouped_term = qml.sum(*(flat_hamiltonian.operands[index] for index in group_indices))
    groups.append(grouped_term)

grouped_hamiltonian = qml.sum(*groups)
t2 = time.time()


t3 = time.time()
n_q = 2 * n_cell**2
n_xx = n_cell**2
n_yy = n_cell*(n_cell-1)
n_zz = n_yy

commuting_groups = [
    {"XX": n_xx},
    {"YY": n_yy},
    {"ZZ": n_zz},
]

compact_hamiltonian = qre.PauliHamiltonian(
    num_qubits = n_q,
    commuting_groups = commuting_groups,
)
t4 = time.time()

print(f"Processing time: ~ {(t2 - t1):.3E} sec")
print("Total number of terms:", len(flat_hamiltonian.operands))
print("Total number of qubits:", len(flat_hamiltonian.wires), "\n")

print(f"Processing time: ~ {(t4 - t3):.3E} sec")
print("Total number of terms:", compact_hamiltonian.num_pauli_words)
print("Total number of qubits:", compact_hamiltonian.num_qubits)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Processing time: ~ 9.193E+01 sec
#    Total number of terms: 2640
#    Total number of qubits: 1800
#    
#    Processing time: ~ 5.877E-04 sec
#    Total number of terms: 2640
#    Total number of qubits: 1800

order = 2
num_trotter_steps = 1

@qml.qnode(qml.device("default.qubit"))
def executable_circuit(hamiltonian):
    
    for wire in hamiltonian.wires:  # Uniform State prep
        qml.Hadamard(wire)

    qml.TrotterProduct(hamiltonian, time=1.0, n=num_trotter_steps, order=order)
    return qml.state()

def circuit(hamiltonian):
    qre.UniformStatePrep(num_states = 2**n_q)    # Prepare a uniform superposition over all 2^num_qubit basis states
    qre.TrotterPauli(hamiltonian, num_trotter_steps, order)
    return

t5 = time.time()
resources_exec = qre.estimate(executable_circuit)(grouped_hamiltonian)
t6 = time.time()

print(f"Processing time: ~ {(t6 - t5):.3E} sec")
print(resources_exec)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Processing time: ~ 1.734E+01 sec
#    --- Resources: ---
#     Total wires: 1800
#       algorithmic wires: 1800
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.151E+5
#       'T': 1.940E+5,
#       'CNOT': 8.820E+3,
#       'Z': 3.480E+3,
#       'S': 6.960E+3,
#       'Hadamard': 1.800E+3

t5 = time.time()
resources_compact = qre.estimate(circuit)(compact_hamiltonian)
t6 = time.time()

print(f"Processing time: ~ {(t6 - t5):.3E} sec")
print(resources_compact)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Processing time: ~ 4.470E-04 sec
#    --- Resources: ---
#     Total wires: 1800
#       algorithmic wires: 1800
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.151E+5
#       'T': 1.940E+5,
#       'CNOT': 8.820E+3,
#       'Z': 3.480E+3,
#       'S': 6.960E+3,
#       'Hadamard': 1.800E+3