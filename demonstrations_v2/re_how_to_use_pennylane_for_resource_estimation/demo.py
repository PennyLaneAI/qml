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
# `pennylane.estimator <https://docs.pennylane.ai/en/latest/code/qml_estimator.html>`__ is meant to:
# 
# - Make reasoning about quantum algorithms *quick* and painless - no complicated inputs, just tell
#   ``estimator`` what you know.
# - Keep you up to *speed* - ``estimator`` leverages the latest results from the literature to make
#   sure you’re as efficient as can be.
# - Get you moving *even faster* - in the blink of an eye ``estimator`` provides you with resource
#   estimates, and enables effortless customization to enhance your research.
# 
# Let’s import our quantum resource estimator.
# 

import pennylane as qml
import pennylane.estimator as qre

######################################################################
# We will be using the Kitaev model as an example to explore resource estimation. For more information
# about the Kitaev Hamiltonian, check out `our
# documentation <https://docs.pennylane.ai/en/stable/code/api/pennylane.spin.kitaev.html>`__.
# 
# This Hamiltonian is defined through nearest neighbor interactions on a honeycomb shaped lattice as
# follows:
# 
# .. math::
#   \hat{H} = K_X \sum_{\langle i,j \rangle \in X}\sigma_i^x\sigma_j^x +
#   \:\: K_Y \sum_{\langle i,j \rangle \in Y}\sigma_i^y\sigma_j^y +
#   \:\: K_Z \sum_{\langle i,j \rangle \in Z}\sigma_i^z\sigma_j^z
# 
# In this demo we will estimate the quantum resources necessary to evolve the quantum state of a 100 x
# 100 unit honeycomb lattice of spins under the Kitaev Hamiltonian.
# 
# **Thats 20,000 spins!**
# 

import numpy as np
import time

# Construct the Hamiltonian on a 30 units x 30 units lattice
n_cells = [30, 30]
kx, ky, kz = (0.5, 0.6, 0.7)

t1 = time.time()
spin_ham = qml.spin.kitaev(n_cells, coupling=np.array([kx, ky, kz]))
t2 = time.time()

print(f"Generation time: ~ {round(t2 - t1)} sec")
print("Total number of terms:", len(spin_ham.operands))
print("Total number of qubits:", len(spin_ham.wires))

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Generation time: ~ 5 sec
#    Total number of terms: 2640
#    Total number of qubits: 1800

######################################################################
# | It took a few seconds to generate that Hamiltonian. What happens when we are working with even
#   larger systems?
# | Let’s see how this scales.
# 

n_lst = [i for i in range(5,36)]
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
#    Finished n = 5 in ~ 0.01563405990600586 sec
#    Finished n = 10 in ~ 0.08188867568969727 sec
#    Finished n = 15 in ~ 0.43582606315612793 sec
#    Finished n = 20 in ~ 1.0403099060058594 sec
#    Finished n = 25 in ~ 2.650714874267578 sec
#    Finished n = 30 in ~ 5.526278495788574 sec
#    Finished n = 35 in ~ 10.387336492538452 sec

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

power_law = lambda n, a, b: a * n**b

a_fit, b_fit = curve_fit(power_law, n_lst, time_lst, p0=[0.000007, 4])[0]
fit_n = np.concatenate([n_lst, [110]])
projected_n = 100
projected_time = power_law(projected_n, a_fit, b_fit)

plt.plot(n_lst, time_lst, ".", label="Measured generation times")
plt.plot(fit_n, power_law(fit_n, a_fit, b_fit), "--g", 
         label=f"Best-fit")
plt.plot([projected_n], [projected_time], "*r", 
         label=f"n={projected_n}, time: ~{round(projected_time / 60)} mins")
plt.xscale("log"); plt.xlabel("Number of unit cells")
plt.yscale("log"); plt.ylabel("Processing time (sec)")
plt.legend()
plt.show()

######################################################################
#
# .. figure:: ../_static/demonstration_assets/re_how_to_use_pennylane_for_resource_estimation/re_how_to_use_pennylane_for_resource_estimation_a2adf0b7_1.png
#    :align: center
#    :width: 80%

######################################################################
# It would take around 15 minutes just to *generate* the Hamiltonian for a 100 x 100 unit honeycomb
# lattice! Even after that, we would still be stuck with expensive processing tasks.
# 

######################################################################
# Making it Easy
# ~~~~~~~~~~~~~~
# 

######################################################################
# Thanks to ``estimator``, we don’t need a detailed description of our Hamiltonian to estimate its
# resources!
# 
# The geometry of the honeycomb lattice and the structure of the Hamiltonian allows us to calculate
# some important quantities directly:
# 
# .. math::
#   n_{q} = 2 n^{2}, \\
#   n_{YY} = n_{ZZ} = n * (n - 1), \\
#   n_{XX} = n^{2}, \\
# 
# | ``estimator`` provides
#   `classes <https://docs.pennylane.ai/en/latest/code/qml_estimator.html#resource-hamiltonians>`__
#   which allow us to investigate the resources of Hamiltonian simulation without needing to generate
#   them.
# | In this case, we can capture the key information of our Hamiltonian in a compact representation
#   using the ``qre.PauliHamiltonian`` class.
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
# We can then use existing resource
# `operators <https://docs.pennylane.ai/en/latest/code/qml_estimator.html#id1>`__ and
# `templates <https://docs.pennylane.ai/en/latest/code/qml_estimator.html#resource-templates>`__ to
# express our circuit.
# 

order = 2
num_steps = 10

def circuit(hamiltonian):
    qre.UniformStatePrep(num_states = 2**n_q)    # Prepare a uniform superposition over all 2^num_qubit basis states
    qre.TrotterPauli(hamiltonian, num_steps, order)
    return

######################################################################
# So, how do we figure out our quantum resources?
# 
# It’s simple: just call ``qre.estimate``!
# 

t1 = time.time()
res = qre.estimate(circuit)(kitaev_H)
t2 = time.time()

print(f"Processing time: ~ {t2 - t1:.3} sec\n")
print(res)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Processing time: ~ 0.000364 sec
#    
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.862E+7
#       'T': 2.622E+7,
#       'CNOT': 1.192E+6,
#       'Z': 3.960E+5,
#       'S': 7.920E+5,
#       'Hadamard': 2.000E+4

######################################################################
# Our resource estimate was generated in the blink of an eye.
# 
# | We can also analyze the resources of an individual ``ResourceOperator``.
# | Let’s see how the cost of ``qre.TrotterPauli`` changes when we split our terms into groups of
#   commuting terms:
# 

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

# Just compare T gates:
t_count_1 = resources_without_grouping.gate_counts["T"]
t_count_2 = resources_with_grouping.gate_counts["T"]
reduction = abs((t_count_2-t_count_1)/t_count_1)
print("--- With grouping ---", f"\n T gate count: {t_count_1:.3E}\n")
print("--- Without grouping ---", f"\n T gate count: {t_count_2:.3E}\n")
print(f"Difference: {100*reduction:.3}% reduction")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    --- With grouping ---
#     T gate count: 2.622E+07
#    
#    --- Without grouping ---
#     T gate count: 1.791E+07
#    
#    Difference: 31.7% reduction

######################################################################
# By splitting our terms into groups, we’ve managed to reduce the ``T`` gate count of our
# Trotterization by over 30%!
# 

######################################################################
# Gatesets & Configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

######################################################################
# | The cost of an algorithm is typically quantified by the number of logical qubits required and the
#   number of gates used. Different hardware will natively support different gatesets.
# | The default gateset used by ``estimate`` is:
# | ``{'Hadamard', 'S', 'CNOT', 'T', 'Toffoli', 'X', 'Y', 'Z'}``.
# 
# Here are the resources using our updated Hamiltonian, with the default gateset:
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
#     frozenset({'Y', 'S', 'X', 'Hadamard', 'Toffoli', 'T', 'CNOT', 'Z'})
#    
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 1.993E+7
#       'T': 1.791E+7,
#       'CNOT': 8.140E+5,
#       'Z': 3.960E+5,
#       'S': 7.920E+5,
#       'Hadamard': 2.000E+4

######################################################################
# | We can configure the gateset to obtain resource estimates at various levels of abstraction.
# | Here, we configure a high-level gateset which adds gate types such as rotations, and a low
#   level-gateset limited to just ``Hadamard``, ``CNOT``, ``S``, and ``T`` gates.
# 
# We can see how the resources manifest at these different levels.
# 

# Customize gateset: 

highlvl_gateset = {
    'RX', 'RY', 'RZ',
    'Toffoli',
    'X', 'Y', 'Z', 
    'Adjoint(S)', 'Adjoint(T)',
    'Hadamard', 'S', 'CNOT', 'T'
}

highlvl_res = qre.estimate(circuit, gate_set=highlvl_gateset)(kitaev_H_with_grouping)
print(f"High-level resources:\n{highlvl_res}\n")

lowlvl_gateset = {'Hadamard', 'S', 'CNOT', 'T'}

lowlvl_res = qre.estimate(circuit, gate_set=lowlvl_gateset)(kitaev_H_with_grouping)
print(f"Low-level resources:\n{lowlvl_res}")


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
#     Total gates : 2.033E+6
#       'RX': 1.100E+5,
#       'RY': 1.980E+5,
#       'Adjoint(S)': 3.960E+5,
#       'RZ': 9.900E+4,
#       'CNOT': 8.140E+5,
#       'S': 3.960E+5,
#       'Hadamard': 2.000E+4
#    
#    Low-level resources:
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.033E+7
#       'T': 1.791E+7,
#       'CNOT': 8.140E+5,
#       'S': 1.584E+6,
#       'Hadamard': 2.000E+4

######################################################################
# When decomposing our algorithms to a particular gateset, it is often the case that we only have some
# approximate decomposition of our building-block into the target gateset. For example: approximate
# state loading to some precision, or rotation synthesis within some precision of the rotation angle.
# 
# These approximate decompositions are accurate within some error threshold; tuning this error
# threshold determines the resource cost of the algorithm. We can set and tune these errors using a
# resource configuration: ``qre.ResourceConfig``.
# 

custom_rc = qre.ResourceConfig() # generate a resource configuration

rz_precisions = custom_rc.resource_op_precisions[qre.RZ]
print(f"Default setting: {rz_precisions}\n")  # Notice that the default precision for RZ is 1e-9

custom_rc.set_precision(qre.RZ, 1e-15)  # setting the required precision from the default --> 1e-15

res = qre.estimate(
    circuit, 
    gate_set=lowlvl_gateset,
    config=custom_rc,
)(kitaev_H_with_grouping)

# Just compare T gates: 
print("--- Low precision (1e-9) ---", f"\n T counts: {lowlvl_res.gate_counts["T"]:.3E}\n")
print("--- High precision (1e-15) ---", f"\n T counts: {res.gate_counts["T"]:.3E}")     # Notice that a more precise estimate requires more T-gates! 

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    Default setting: {'precision': 1e-09}
#    
#    --- Low precision (1e-9) ---
#     T counts: 1.791E+07
#    
#    --- High precision (1e-15) ---
#     T counts: 2.009E+07

######################################################################
# Swapping Decompositions
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# There are many ways to decompose a quantum gate into our target gateset. Selecting an alternate
# decomposition is a great way to optimize the cost of your quantum workflow. This can be done easily
# with the ``ResourceConfig`` class.
# 
# | Let’s explore decompositions for the ``RZ`` gate:
# | Current decomposition for ``RZ``, or single qubit rotation synthesis in general is: `Efficient
#   Synthesis of Universal Repeat-Until-Success Circuits (Bocharov, et
#   al) <https://arxiv.org/abs/1404.5320>`__
# 

default_cost_RZ = qre.estimate(qre.RZ(precision=1e-9))  # Manually set the precision
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

######################################################################
# These are other state of the art methods we could use instead:
# 
# - `Optimal ancilla-free Cliﬀord+T approximation of z-rotations (Ross,
#   Selinger) <https://arxiv.org/pdf/1403.2975>`__
# - `Shorter quantum circuits via single-qubit gate approximation (Kliuchnikov et
#   al) <https://arxiv.org/abs/2203.10064v2>`__
# 

# According to paper by Ross & Selinger, we can decompose RZ rotations into T-gates according to: 
def gridsynth_t_cost(error):
    return round(3 * qml.math.log2(1/error))

# According to paper by Kliuchnikov et al, we can decompose RZ rotations into T-gates according to: 
def mixed_fallback_t_cost(error):
    return round(0.56 * qml.math.log2(1/error) + 5.3)

######################################################################
# In order to define a resource decomposition, we first need to know the resource_keys for the
# operator whose decomposition we want to add:
# 

print(qre.RZ.resource_keys)  # this tells us all of the REQUIRED arguments our function must take

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    {'precision'}

######################################################################
# Now that we know which arguments we need, we can define our resource decomposition.
# 

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

######################################################################
# Finally, we set the new decomposition in our ``ResourceConfig``.
# 

grisynth_rc = qre.ResourceConfig()
grisynth_rc.set_decomp(qre.RZ, gridsynth_decomp)
grisynth_cost_RZ = qre.estimate(qre.RZ(precision=1e-9), config=grisynth_rc)

mixed_fallback_rc = qre.ResourceConfig()
mixed_fallback_rc.set_decomp(qre.RZ, mixed_fallback_decomp)
mixed_fallback_cost_RZ = qre.estimate(qre.RZ(precision=1e-9), config=mixed_fallback_rc)

print("GridSynth decomposition -", f"\tT count: {grisynth_cost_RZ.gate_counts["T"]}")
print("Default decomposition (RUS) -", f"\tT count: {default_cost_RZ.gate_counts["T"]}")
print("Mixed Fallback decomposition -", f"\tT count: {mixed_fallback_cost_RZ.gate_counts["T"]}")

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
# .. code-block:: none
# 
#    GridSynth decomposition - 	T count: 90
#    Default decomposition (RUS) - 	T count: 44
#    Mixed Fallback decomposition - 	T count: 22

######################################################################
# Putting it All Together
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# We can combine all of the features we have seen so far to optimize the cost of Trotterized time
# evolution of the Kitaev hamiltonian:
# 

kitaev_hamiltonian = kitaev_H_with_grouping  # use compact hamiltonian with grouping

custom_gateset = lowlvl_gateset # use the low-level gateset

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
#     Total gates : 1.874E+7
#       'T': 1.632E+7,
#       'CNOT': 8.140E+5,
#       'S': 1.584E+6,
#       'Hadamard': 2.000E+4

######################################################################
# Estimating the Resources of your PennyLane Circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

######################################################################
# Your turn!
# ~~~~~~~~~~
# 
# Now that you’ve seen how powerful PennyLane’s ``estimator`` is, go `try it out
# yourself <https://docs.pennylane.ai/en/latest/code/qml_estimator.html>`__!
# 
# Reason about the costs of your quantum algorithm without any of the headaches.
# 