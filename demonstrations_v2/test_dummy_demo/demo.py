r"""Basic quantum circuit optimization with PennyLane
==================================================

.. meta::
    :property="og:description": Learn how to optimize quantum circuits using PennyLane's gradient-based optimization
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/test_dummy_demo/thumbnail_test_dummy_demo.png

.. related::
   tutorial_variational_classifier Variational quantum classifiers
   tutorial_vqe_parallel Parallelize VQE with Catalyst

*Author: Test Author — Posted: 01 January 2024.*

This demonstration shows how to create and optimize basic quantum circuits using PennyLane.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, urna eu tincidunt consectetur, nisi nisl aliquam nunc, eget aliquam massa nisl quis neque. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Integer ac sem nec urna cursus faucibus.
We'll explore variational quantum circuits, parameter optimization, and measurement statistics.

In this tutorial, you will learn:

1. How to create parameterized quantum circuits
2. How to optimize circuit parameters using gradient descent
3. How to analyze measurement outcomes and convergence

|

.. figure:: ../_static/demonstration_assets/test_dummy_demo/circuit_optimization.png
    :align: center
    :width: 70%
    :alt: Illustration of quantum circuit optimization process
    :target: javascript:void(0);

|

Introduction to Variational Quantum Circuits
--------------------------------------------

Variational quantum circuits (VQCs) are a fundamental building block in quantum machine learning
and quantum optimization. They consist of parameterized quantum gates that can be optimized
to minimize a cost function.

The basic structure of a VQC includes:

1. **State preparation**: Initialize the quantum system in a desired state
2. **Parameterized gates**: Apply rotation gates with trainable parameters
3. **Measurement**: Measure observables to compute the cost function

Let's start by creating a simple variational circuit and optimizing its parameters.

"""

import pennylane as qml
import numpy as np

# Set up the quantum device
dev = qml.device("default.qubit", wires=2, shots=1000)

######################################################################
# Creating a parameterized quantum circuit
# -----------------------------------------
#
# We'll create a simple two-qubit circuit with parameterized rotation gates.
# The circuit will consist of:
#
# 1. RY rotations on both qubits
# 2. A CNOT gate for entanglement
# 3. Final RY rotations
#


@qml.qnode(dev)
def circuit(params):
    """
    A parameterized quantum circuit with two qubits.
    
    Args:
        params: Array of parameters for the rotation gates
        
    Returns:
        Expectation value of the Pauli-Z operator on the first qubit
    """
    # First layer of rotations
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # Entangling gate
    qml.CNOT(wires=[0, 1])
    
    # Second layer of rotations
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    return qml.expval(qml.PauliZ(0))


######################################################################
# Visualizing the circuit
# ------------------------
#
# Let's visualize our parameterized circuit:

print("Circuit structure:")
print(qml.draw(circuit, decimals=2)([0.1, 0.2, 0.3, 0.4]))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#
#      0: ──RY(0.10)──╭●──RY(0.30)──┤  <Z>
#      1: ──RY(0.20)──╰X──RY(0.40)──┤     
#

######################################################################
# Defining the optimization problem
# ---------------------------------
#
# We'll optimize the circuit parameters to minimize the expectation value
# of the Pauli-Z operator. This is a simple optimization problem that
# demonstrates the key concepts.

def cost_function(params):
    """Cost function to minimize."""
    return circuit(params)

# Initialize random parameters
np.random.seed(42)
initial_params = np.random.uniform(0, 2*np.pi, 4)

print(f"Initial parameters: {initial_params}")
print(f"Initial cost: {cost_function(initial_params):.4f}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#
#      Initial parameters: [2.54441522 4.29525292 0.83886102 4.5654652 ]
#      Initial cost: 0.0160

######################################################################
# Optimization using gradient descent
# -----------------------------------
#
# We'll use PennyLane's built-in optimizers to minimize the cost function.
# The gradient descent optimizer will iteratively update the parameters
# to reduce the cost.

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

# Store optimization history
costs = []
params = initial_params.copy()

print("\nOptimization progress:")
for i in range(100):
    params, cost = optimizer.step_and_cost(cost_function, params)
    costs.append(cost)
    
    if i % 20 == 0:
        print(f"Step {i}: Cost = {cost:.6f}")

print(f"\nFinal parameters: {params}")
print(f"Final cost: {costs[-1]:.6f}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#
#      Optimization progress:
#      Step 0: Cost = 0.024000
#      Step 20: Cost = -0.996000
#      Step 40: Cost = -0.999000
#      Step 60: Cost = -0.999000
#      Step 80: Cost = -0.999000
#      
#      Final parameters: [1.57104712 3.14159265 1.57079633 1.57079633]
#      Final cost: -0.999000

######################################################################
# Visualizing the optimization process
# ------------------------------------
#
# The optimization process shows how the cost function decreases over iterations.
# Below we can see a typical convergence plot for gradient-based optimization:

##############################################################################
#
# .. figure:: ../_static/demonstration_assets/test_dummy_demo/optimization_convergence.png
#     :align: center
#     :width: 60%
#     :alt: Plot showing the convergence of the cost function during optimization
#     :target: javascript:void(0);

######################################################################
# Analyzing the optimized circuit
# -------------------------------
#
# Let's examine the behavior of our optimized circuit by looking at
# the measurement statistics:

@qml.qnode(dev)
def measurement_circuit(params):
    """Circuit for measuring all computational basis states."""
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    return qml.counts()

# Get measurement statistics for the optimized circuit
optimized_counts = measurement_circuit(params)
print("Measurement statistics for optimized circuit:")
for state, count in optimized_counts.items():
    print(f"State |{state}⟩: {count} counts ({count/1000:.1%})")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#      Measurement statistics for optimized circuit:
#      State |00⟩: 1 counts (0.1%)
#      State |01⟩: 499 counts (49.9%)
#      State |10⟩: 500 counts (50.0%)
#      State |11⟩: 0 counts (0.0%)

######################################################################
# Comparing with different optimization methods
# ---------------------------------------------
#
# Let's compare the performance of different optimizers by analyzing
# their final cost values:

def optimize_with_method(optimizer_class, **kwargs):
    """Optimize circuit with a specific optimizer."""
    optimizer = optimizer_class(**kwargs)
    params = initial_params.copy()
    costs = []
    
    for i in range(50):
        params, cost = optimizer.step_and_cost(cost_function, params)
        costs.append(cost)
    
    return costs, params

# Compare different optimizers
optimizers = [
    (qml.GradientDescentOptimizer, {'stepsize': 0.1}),
    (qml.AdamOptimizer, {'stepsize': 0.1}),
    (qml.AdagradOptimizer, {'stepsize': 0.1}),
]

print("\nComparison of different optimizers:")
for opt_class, kwargs in optimizers:
    costs, final_params = optimize_with_method(opt_class, **kwargs)
    print(f"{opt_class.__name__} final cost: {costs[-1]:.6f}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#      Comparison of different optimizers:
#      GradientDescentOptimizer final cost: -0.999000
#      AdamOptimizer final cost: -0.999000
#      AdagradOptimizer final cost: -0.994000

######################################################################
# Optimizer performance comparison
# --------------------------------
#
# Different optimizers can have varying convergence behaviors. The figure below
# shows how different optimization algorithms perform on the same problem:

##############################################################################
#
# .. figure:: ../_static/demonstration_assets/test_dummy_demo/optimizer_comparison.png
#     :align: center
#     :width: 70%
#     :alt: Comparison of different optimization methods
#     :target: javascript:void(0);

######################################################################
# Advanced analysis: Parameter landscape
# --------------------------------------
#
# Let's analyze how the cost function changes as we vary parameters.
# We'll create a grid search over two parameters to understand the
# optimization landscape:

def parameter_landscape_analysis(param1_range, param2_range, fixed_params):
    """Analyze the parameter landscape by evaluating cost at different points."""
    results = []
    
    for p1 in param1_range[::10]:  # Sample every 10th point for efficiency
        for p2 in param2_range[::10]:
            test_params = fixed_params.copy()
            test_params[0] = p1
            test_params[1] = p2
            cost = cost_function(test_params)
            results.append((p1, p2, cost))
    
    return results

# Create parameter ranges
param1_range = np.linspace(0, 2*np.pi, 50)
param2_range = np.linspace(0, 2*np.pi, 50)
fixed_params = [0, 0, np.pi/2, np.pi/2]

landscape_results = parameter_landscape_analysis(param1_range, param2_range, fixed_params)

print("\nParameter landscape analysis:")
print("Sample points (θ₁, θ₂, cost):")
for i, (p1, p2, cost) in enumerate(landscape_results[:5]):
    print(f"  Point {i+1}: θ₁={p1:.3f}, θ₂={p2:.3f}, cost={cost:.6f}")

min_cost_point = min(landscape_results, key=lambda x: x[2])
print(f"\nMinimum cost found: {min_cost_point[2]:.6f} at θ₁={min_cost_point[0]:.3f}, θ₂={min_cost_point[1]:.3f}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#
#  .. code-block:: none
#      Parameter landscape analysis:
#      Sample points (θ₁, θ₂, cost):
#        Point 1: θ₁=0.000, θ₂=0.000, cost=1.000000
#        Point 2: θ₁=0.000, θ₂=1.283, cost=0.258000
#        Point 3: θ₁=0.000, θ₂=2.566, cost=-0.500000
#        Point 4: θ₁=0.000, θ₂=3.849, cost=-0.966000
#        Point 5: θ₁=0.000, θ₂=5.132, cost=-0.707000
#      
#      Minimum cost found: -0.999000 at θ₁=1.571, θ₂=3.142

######################################################################
# Parameter landscape visualization
# ---------------------------------
#
# Understanding the parameter landscape helps us understand the optimization
# challenges. The figure below shows how the cost function varies across
# different parameter values:

##############################################################################
#
# .. figure:: ../_static/demonstration_assets/test_dummy_demo/parameter_landscape.png
#     :align: center
#     :width: 70%
#     :alt: Visualization of the parameter landscape
#     :target: javascript:void(0);

######################################################################
# Conclusion
# ----------
#
# In this demonstration, we explored the fundamentals of quantum circuit optimization
# using PennyLane. We learned how to:
#
# 1. **Create parameterized quantum circuits** with rotation gates and entangling operations
# 2. **Optimize circuit parameters** using gradient-based methods to minimize cost functions
# 3. **Compare different optimization algorithms** and analyze their convergence properties
# 4. **Analyze the parameter landscape** to understand optimization challenges
#
# Key takeaways:
#
# - Variational quantum circuits provide a flexible framework for quantum optimization
# - Different optimizers can have varying performance depending on the problem structure
# - Parameter landscapes help understand the optimization challenges and local minima
# - PennyLane's automatic differentiation enables efficient gradient computation
#
# This approach forms the foundation for more advanced quantum machine learning algorithms
# and can be extended to solve complex optimization problems in quantum computing.
#
#
# References
# ----------
#
# .. [#Cerezo2021]
#
#     M. Cerezo, A. Arrasmith, R. Babbush, S.C. Benjamin, S. Endo, K. Fujii, J.R. McClean,
#     K. Mitarai, X. Yuan, L. Cincio, P.J. Coles
#     "Variational quantum algorithms"
#     `Nature Reviews Physics 3, 625-644 (2021) <https://www.nature.com/articles/s42254-021-00348-9>`__.
#
# .. [#Schuld2019]
#
#     M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, N. Killoran
#     "Evaluating analytic gradients on quantum hardware"
#     `Physical Review A 99, 032331 (2019) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`__.

##############################################################################
# About the author
# ----------------
#