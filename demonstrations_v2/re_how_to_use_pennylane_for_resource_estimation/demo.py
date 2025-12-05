r"""
How to use PennyLane for Resource Estimation
============================================
"""

######################################################################
# Fault tolerant quantum computers are on their way.
# But how do we ensure that useful algorithms can actually run on them?
# An algorithm is hardly helpful when it cannot be executed;
# but only truly helpful when it can.
#
# This is a major challenge in quantum algorithm development,
# especially since we are often working at scales where simulation is no longer feasible.
# We therefore need to analyze our algorithms to perform **resource estimation**:
# getting an idea of how many resources an algorithm requires, such as logical qubits and gates.
# In turn, this gives us an indication of how long the algorithm will take to execute
# on a given quantum hardware architecture,
# or if it will even fit in memory to begin with.
#
# PennyLane is here to make that process easy, with our new resource estimation module:
# :mod:`estimator <pennylane.estimator>`.
#
# In this demo, we will show you how to perform resource estimation
# for a simple Hamiltonian simulation workflow.
#
# Let’s import our quantum resource :mod:`estimator <pennylane.estimator>`.
#

import pennylane as qml
import pennylane.estimator as qre

######################################################################
#
# PennyLane's :mod:`estimator <pennylane.estimator>` module:
#
# - Makes reasoning about quantum algorithms *quick* and painless - no complicated inputs, just tell
#   :mod:`estimator <pennylane.estimator>` what you know.
# - Keeps you up to *speed* - :mod:`estimator <pennylane.estimator>` leverages the latest results from the literature to make
#   sure you’re as efficient as can be.
# - Gets you moving *even faster* - in the blink of an eye :mod:`estimator <pennylane.estimator>`
#   provides you with resource estimates, and enables effortless customization to enhance your research.
#
# We will estimate the quantum resources necessary to evolve the quantum state of a
# honeycomb lattice of spins under the Kitaev Hamiltonian.
# For more information about the Kitaev Hamiltonian,
# check out :func:`our documentation <pennylane.spin.kitaev>`.
#

######################################################################
# Estimating the Resources of your PennyLane Circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's say you've already written your workflow as a PennyLane circuit.
# There's no need to write it again!
# To estimate its resources, you can simply invoke
# :func:`qre.estimate <pennylane.estimator.estimate.estimate>`
# on it directly.
#
# Let's demonstrate this with a 25 x 25 unit honeycomb lattice of spins.  
# Here, we generate the Hamiltonian ourselves:
#

import numpy as np

n_cell = 25
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
t_generation = t2 - t1

print(f"Processing time for Hamiltonian generation: {(t_generation):.3g} seconds")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time for Hamiltonian generation: 7.74 seconds

######################################################################
# Here we define our circuit for Hamiltonian simulation.
#

num_steps = 10
order = 6

@qml.qnode(qml.device("default.qubit"))
def executable_circuit(hamiltonian, num_steps, order):
    for wire in hamiltonian.wires: # uniform superposition over all basis states
        qml.Hadamard(wire)
    qml.TrotterProduct(hamiltonian, time=1.0, n=num_steps, order=order)
    return qml.state()

######################################################################
# Now, just call :func:`qre.estimate <pennylane.estimator.estimate.estimate>`
# to generate a state-of-the-art resource estimate for your PennyLane circuit!
#

t1 = time.time()
resources_exec = qre.estimate(executable_circuit)(grouped_hamiltonian, num_steps, order)
t2 = time.time()

print(f"Processing time: {(t2 - t1):.3g} seconds")
print(resources_exec)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time: 2.32 seconds
#    --- Resources: ---
#     Total wires: 1250
#       algorithmic wires: 1250
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.972E+7
#       'T': 2.670E+7,
#       'CNOT': 1.214E+6,
#       'Z': 6.000E+5,
#       'S': 1.200E+6,
#       'Hadamard': 1.250E+3

######################################################################
# Making it Easy
# ~~~~~~~~~~~~~~
#

######################################################################
# What if we wanted to estimate the quantum resources necessary to evolve the quantum state of a 100 x
# 100 unit honeycomb lattice of spins under the Kitaev Hamiltonian?
#
# **Thats 20,000 spins!**
#
# Generating such Hamiltonians quickly becomes a bottleneck.
# However, thanks to :mod:`estimator <pennylane.estimator>`,
# we don’t need a detailed description of our Hamiltonian to estimate our algorithm's resources!
#
# The geometry of the honeycomb lattice and the structure of the Hamiltonian allow us to calculate
# some important quantities directly:
#
# .. math::
#   n_{q} &= 2 n^{2}, \\
#   n_{YY} &= n_{ZZ} = n * (n - 1), \\
#   n_{XX} &= n^{2}, \\
#
# Our quantum resource :mod:`estimator <pennylane.estimator>` provides
# `classes <https://docs.pennylane.ai/en/stable/code/qml_estimator.html#resource-hamiltonians>`__
# which allow us to investigate the resources of Hamiltonian simulation without needing to generate
# them.
# In this case, we can capture the key information of our Hamiltonian in a compact representation
# using the
# :class:`qre.PauliHamiltonian <pennylane.estimator.compact_hamiltonian.PauliHamiltonian>`
# class.
#

n_cell = 100
n_q = 2 * n_cell**2
n_xx = n_cell**2
n_yy = n_cell * (n_cell - 1)
n_zz = n_yy

pauli_word_distribution = {"XX": n_xx, "YY": n_yy, "ZZ": n_zz}

kitaev_H = qre.PauliHamiltonian(
    num_qubits=n_q,
    pauli_dist=pauli_word_distribution,
)

######################################################################
# We can then use existing resource
# `operators <https://docs.pennylane.ai/en/stable/code/qml_estimator.html#id1>`__ and
# `templates <https://docs.pennylane.ai/en/stable/code/qml_estimator.html#resource-templates>`__
# from the :mod:`estimator <pennylane.estimator>` module to express our circuit.
# These
# :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`
# classes are designed to require minimal information
# while still providing trustworthy estimates.
#

def circuit(hamiltonian, num_steps, order):
    qre.UniformStatePrep(num_states=2**n_q)  # uniform superposition over all basis states
    qre.TrotterPauli(hamiltonian, num_steps, order)

######################################################################
# The cost of an algorithm is typically quantified by the number of logical qubits required and the
# number of gates used. Different hardware may natively support different gatesets.
# The default gateset used by :mod:`estimator <pennylane.estimator>` is:
# ``{'Hadamard', 'S', 'CNOT', 'T', 'Toffoli', 'X', 'Y', 'Z'}``.
#
# So, how do we figure out our quantum resources?
#
# It’s simple: just call :func:`qre.estimate <pennylane.estimator.estimate.estimate>`!
#

import time

t1 = time.time()
res = qre.estimate(circuit)(kitaev_H, num_steps, order)
t2 = time.time()

print(f"Processing time: {t2 - t1:.3g} seconds\n")
print(res)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time: 0.00213 seconds
#
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 7.151E+8
#       'T': 6.556E+8,
#       'CNOT': 2.980E+7,
#       'Z': 9.900E+6,
#       'S': 1.980E+7,
#       'Hadamard': 2.000E+4

######################################################################
# Our resource estimate was generated in the blink of an eye.
#
# We can also analyze the resources of an individual
# :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`.
# This can be helpful in determining which operators in a workflow demand the most resources.

resources_without_grouping = qre.estimate(qre.TrotterPauli(kitaev_H, num_steps, order))

######################################################################
# Providing additional information can help to produce more accurate resource estimates.
# In the case of our
# :class:`qre.PauliHamiltonian <pennylane.estimator.compact_hamiltonian.PauliHamiltonian>`,
# we can split the terms into groups of commuting terms:

commuting_groups = [{"XX": n_xx}, {"YY": n_yy}, {"ZZ": n_zz}]

kitaev_H_with_grouping = qre.PauliHamiltonian(
    num_qubits=n_q,
    commuting_groups=commuting_groups,
)

resources_with_grouping = qre.estimate(
    qre.TrotterPauli(kitaev_H_with_grouping, num_steps, order)
)

######################################################################
# Let’s see how the cost of ``qre.TrotterPauli`` differs in these two cases!

# Just compare T gates:
t_count_1 = resources_without_grouping.gate_counts["T"]
t_count_2 = resources_with_grouping.gate_counts["T"]
reduction = abs((t_count_2 - t_count_1) / t_count_1)
print("--- With grouping ---", f"\n T gate count: {t_count_1:.3E}\n")
print("--- Without grouping ---", f"\n T gate count: {t_count_2:.3E}\n")
print(f"Difference: {100*reduction:.1f}% reduction")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    --- With grouping --- 
#     T gate count: 6.556E+08
#   
#    --- Without grouping --- 
#     T gate count: 4.371E+08
#   
#    Difference: 33.3% reduction

######################################################################
# By splitting our terms into groups, we’ve managed to reduce the ``T`` gate count of our
# Trotterization by over 30 percent!
#

######################################################################
# Gatesets & Configurations
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# Here are the resources for our entire circuit using our updated Hamiltonian:

res = qre.estimate(circuit)(kitaev_H_with_grouping, num_steps, order)
print(f"\n{res}")

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
#     Total gates : 4.867E+8
#       'T': 4.371E+8,
#       'CNOT': 1.987E+7,
#       'Z': 9.900E+6,
#       'S': 1.980E+7,
#       'Hadamard': 2.000E+4

######################################################################
# We can configure the gateset to obtain resource estimates at various levels of abstraction.
# Here, we configure a high-level gateset which adds gate types such as rotations, and a low
# level-gateset limited to just ``Hadamard``, ``CNOT``, ``S``, and ``T`` gates.
#
# We can see how the resources manifest at these different levels.
#

highlvl_gateset = {
    "RX","RY","RZ",
    "Toffoli",
    "X","Y","Z",
    "Adjoint(S)","Adjoint(T)",
    "Hadamard","S","CNOT","T",
}

highlvl_res = qre.estimate(
    circuit,
    gate_set=highlvl_gateset,
)(kitaev_H_with_grouping, num_steps, order)

print(f"High-level resources:\n{highlvl_res}\n")

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
#     Total gates : 4.962E+7
#       'RX': 2.510E+6,
#       'RY': 4.950E+6,
#       'Adjoint(S)': 9.900E+6,
#       'RZ': 2.475E+6,
#       'CNOT': 1.987E+7,
#       'S': 9.900E+6,
#       'Hadamard': 2.000E+4

lowlvl_gateset = {"Hadamard", "S", "CNOT", "T"}

lowlvl_res = qre.estimate(
    circuit,
    gate_set=lowlvl_gateset,
)(kitaev_H_with_grouping, num_steps, order)

print(f"Low-level resources:\n{lowlvl_res}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Low-level resources:
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 4.966E+8
#       'T': 4.371E+8,
#       'CNOT': 1.987E+7,
#       'S': 3.960E+7,
#       'Hadamard': 2.000E+4

######################################################################
# When decomposing our algorithms to a particular gateset, it is often the case that we only have some
# approximate decomposition of a building-block into the target gateset. For example, approximate
# state loading to some precision, or rotation synthesis within some precision of the rotation angle.
#
# These approximate decompositions are accurate within some error threshold; tuning this error
# threshold impacts the required resources. We can set and tune these errors using a
# resource configuration: :class:`ResourceConfig <pennylane.estimator.resource_config.ResourceConfig>`.
#
# Notice that a more precise estimate requires more ``T`` gates!
#

custom_rc = qre.ResourceConfig()  # generate a resource configuration

rz_precisions = custom_rc.resource_op_precisions[qre.RZ]
print(f"Default setting: {rz_precisions}\n")

custom_rc.set_precision(qre.RZ, 1e-15) # customize precision

res = qre.estimate(
    circuit,
    gate_set=lowlvl_gateset,
    config=custom_rc, # provide our custom configuration
)(kitaev_H_with_grouping, num_steps, order)

# Just compare T gates:
print("--- Lower precision (1e-9) ---", f"\n T counts: {lowlvl_res.gate_counts["T"]:.3E}")
print("\n--- Higher precision (1e-15) ---", f"\n T counts: {res.gate_counts["T"]:.3E}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Default setting: {'precision': 1e-09}
#    
#    --- Lower precision (1e-9) --- 
#     T counts: 4.371E+08
#    
#    --- Higher precision (1e-15) --- 
#     T counts: 4.916E+08

######################################################################
# The :mod:`estimator <pennylane.estimator>` module also provides functionality for
# writing custom decompositions and custom resource operators.
# To find out how, check out our documentation for
# :class:`ResourceConfig <pennylane.estimator.resource_config.ResourceConfig>`
# and :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`!

######################################################################
# Putting it All Together
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We can combine all of the features we have seen so far to determine
# the cost of Trotterized time evolution of the Kitaev Hamiltonian
# in our preferred setting:
#

t1 = time.time()

kitaev_hamiltonian = kitaev_H_with_grouping  # use compact Hamiltonian with grouping

custom_gateset = lowlvl_gateset # use the low-level gateset

custom_config = qre.ResourceConfig()
custom_config.set_precision(qre.RZ, precision=1e-12) # set higher precision

resources = qre.estimate(
    circuit,
    gate_set = custom_gateset,
    config = custom_config
)(kitaev_hamiltonian, num_steps, order)

t2 = time.time()
print(f"Processing time: {t2 - t1:.3g} seconds\n")
print(resources)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time: 0.000425 seconds
#    
#    --- Resources: ---
#     Total wires: 2.000E+4
#       algorithmic wires: 20000
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 5.239E+8
#       'T': 4.644E+8,
#       'CNOT': 1.987E+7,
#       'S': 3.960E+7,
#       'Hadamard': 2.000E+4

######################################################################
# A Final Comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We've shown that you can estimate your workflow's resources
# using both typical PennyLane circuits, and circuits written with
# :class:`ResourceOperator <pennylane.estimator.resource_operator.ResourceOperator>`s.
#
# Now, we'll demonstrate that the resource estimates are consistent across both of these cases!
#
# Let's return to a 25 x 25 unit honeycomb lattice of spins.  
# We'll use :mod:`estimator <pennylane.estimator>` to make sure everything matches.
# It's a lot easier to prepare for resource estimation than for execution!
#

t1 = time.time()
n_cell = 25
n_q = 2 * n_cell**2
n_xx = n_cell**2
n_yy = n_cell*(n_cell-1)
n_zz = n_yy

commuting_groups = [{"XX": n_xx}, {"YY": n_yy}, {"ZZ": n_zz}]

compact_hamiltonian = qre.PauliHamiltonian(
    num_qubits = n_q,
    commuting_groups = commuting_groups,
)
t2 = time.time()
t_estimation = t2 - t1

######################################################################
# The resulting data can be easily compared for a sanity check.
#

print(f"Processing time for Hamiltonian generation: {(t_generation):.3g} seconds")
print("Total number of terms:", len(flat_hamiltonian.operands))
print("Total number of qubits:", len(flat_hamiltonian.wires), "\n")

print(f"Processing time for Hamiltonian estimation: {(t_estimation):.3g} seconds")
print("Total number of terms:", compact_hamiltonian.num_pauli_words)
print("Total number of qubits:", compact_hamiltonian.num_qubits)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time for Hamiltonian generation: 7.74 seconds
#    Total number of terms: 1825
#    Total number of qubits: 1250 
#    
#    Processing time for Hamiltonian estimation: 0.000191 seconds
#    Total number of terms: 1825
#    Total number of qubits: 1250

######################################################################
# Here's the resource estimate from our earlier *execution* circuit.
#

print(resources_exec)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time: 2.32 seconds
#    --- Resources: ---
#     Total wires: 1250
#       algorithmic wires: 1250
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.972E+7
#       'T': 2.670E+7,
#       'CNOT': 1.214E+6,
#       'Z': 6.000E+5,
#       'S': 1.200E+6,
#       'Hadamard': 1.250E+3

######################################################################
# Let's validate the results by comparing with our resource *estimation* circuit.
#

t1 = time.time()
resources_est = qre.estimate(circuit)(compact_hamiltonian, num_steps, order)
t2 = time.time()

print(f"Processing time: {(t2 - t1):.3g} seconds")
print(resources_est)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
# .. code-block:: none
#
#    Processing time: 0.000371 seconds
#    --- Resources: ---
#     Total wires: 1250
#       algorithmic wires: 1250
#       allocated wires: 0
#         zero state: 0
#         any state: 0
#     Total gates : 2.972E+7
#       'T': 2.670E+7,
#       'CNOT': 1.214E+6,
#       'Z': 6.000E+5,
#       'S': 1.200E+6,
#       'Hadamard': 1.250E+3

######################################################################
# The numbers check out!
#

######################################################################
# Your turn!
# ~~~~~~~~~~
#
# Now that you’ve seen how powerful PennyLane’s
# quantum resource :mod:`estimator <pennylane.estimator>` is,
# go try it out yourself!
#
# Use PennyLane to reason about the costs of your quantum algorithm
# without any of the headaches.
#
# References
# ----------
#
# .. [#bocharov]
#
#    Bocharov, Alex, Martin Roetteler, and Krysta M. Svore. 
#    "Efficient synthesis of universal repeat-until-success quantum circuits."
#    Physical review letters 114.8 (2015): 080502. `arXiv <https://arxiv.org/abs/1404.5320>`__.
#
# .. [#ross]
#
#    Ross, Neil J., and Peter Selinger. 
#    "Optimal ancilla-free Clifford+T approximation of z-rotations."
#    Quantum information and computation 16.11–12 (2016): 901–953. `arXiv <https://arxiv.org/abs/1403.2975>`__.
#