"""
The Quadratic Unconstrained Binary Optimization (QUBO)
======================================================

Solving combinatorial optimization problems using quantum computing is
one of those promising applications for the near term. But, why are
combinatorial optimization problems even important? Well, we care about
them because we have useful applications that can be translated into
combinatorial optimization problems, in fields such as logistics,
finance, and engineering. But having useful applications is not enough,
and it is here where the second ingredient comes in, combinatorial
optimization problems are complex! and finding good solutions
(classically) for large instances of them requires an enormous amount of
computational resources.

In this demo, we will be using a quantum algorithm called the Quantum
Approximate Optimization Algorithm (QAOA) to solve a combinatorial
optimization problem. QAOA is an algorithm that leverages the power of
quantum computers to find the ground state of a target Hamiltonian. We
will show throughout this demo how to encode the Knapsack as a target
Hamiltonian and solve it using QAOA.

Table of Contents
=================

1. `Generalities: Quadratic unconstrained binary optimization <#qubo>`__
2. `The Knapsack problem <#KP>`__\  2.1 `Example <#example>`__\  2.2
   `Ising Hamiltonian <#ising>`__\  2.3 `Brute force
   solution <#brute_sol>`__
3. `QAOA <#qaoa>`__\  3.1 `QAOA circuit <#qaoa_circ>`__\  3.2
   `Optimization <#qaoa_opt>`__\  3.3 `Visualization <#visualization>`__
4. `Task 3 <#task3>`__

"""


######################################################################
#  1. Generalities: Quadratic unconstrained binary optimization (QUBO)
# --------------------------------------------------------------------
#
# The set of combinatorial problems that can be represented by the QUBO
# formulation is characterized by functions of the form
#
# :raw-latex:`\begin{equation}
# f(\mathrm{x}) = \frac{1}{2}\sum_{i=1}^{n} \sum_{j=1}^n q_{ij} x_{i} x_{j}, \tag{1}
# \end{equation}` where :math:`n` is the number of variables,
# :math:`q_{ij} \in \mathbb{R}` are coefficients associated to the
# specific problem, and :math:`x_i \in \{0,1\}` are the binary variables
# of the problem. Note that :math:`x_{i} x_{i} \equiv x_{i}` and
# :math:`q_{ij} = q_{ji}` in this formulation. Therefore, the general form
# of a combinatorial optimization problem solvable by QPUs is given by the
# cost function
#
# :raw-latex:`\begin{equation}\label{QUBO_form}
# f(\mathrm{x}) = \sum_{i=1}^{n-1} \sum_{j > i}^n q_{ij}x_{i}x_{j} + \frac{1}{2}\sum_{i=1}^n q_{ii} x_i,\tag{2}
# \end{equation}` and equality constraints are given by
#
# :raw-latex:`\begin{equation}
# \sum_{i=1}^n c_i x_i = C, \ c_i \in \mathbb{Z}, \tag{3}
# \end{equation}` and inequality constraints are given by
#
# :raw-latex:`\begin{equation}\label{inequality}
# \sum_{i=1}^n l_i x_i \le B, \ l_i \in \mathbb{Z} \tag{4}
# \end{equation}`
#
# where :math:`C` and :math:`B` are constants. To transform these problems
# into the QUBO formulation the constraints are added as penalization
# terms. In this respect, the equality constraints are included in the
# cost function using the following penalization term
#
# :raw-latex:`\begin{equation}\label{EQ_F}
# \lambda_0 \left(\sum_{i=1}^n c_i x_i - C\right)^2,\tag{5}
# \end{equation}` where :math:`\lambda_0` is a penalization coefficient
# that should be chosen to guarantee that the equality constraint is
# fulfilled. In the case of inequality constraint, the common approach is
# to use a `slack
# variable <https://en.wikipedia.org/wiki/Slack_variable#:~:text=In%20an%20optimization%20problem%2C%20a,constraint%20on%20the%20slack%20variable.>`__.
# The slack variable, :math:`S`, is an auxiliary variable that makes a
# penalization term vanish when the inequality constraint is achieved,
#
# :raw-latex:`\begin{equation}\label{ineq}
#  \sum_{i=1}^n B - l_i x_i - S = 0.\tag{6}
#  \end{equation}` Therefore, when Eq.(:raw-latex:`\ref{inequality}`) is
# satisfied, Eq.(:raw-latex:`\ref{ineq}`) is already zero. This means the
# slack variable, :math:`S`, must be in the range
# :math:`0 \le S \le \max_x \sum_{i=1}^n B - l_i x_i`. To represent the
# :math:`slack` variable in binary form, the slack is decomposed in binary
# variables:
#
# :raw-latex:`\begin{equation}\label{SB}
# S = \sum_{k=0}^{N-1} 2^k s_k,\tag{7}
# \end{equation}` where :math:`s_k` are the slack binary variables. Then,
# the inequality constraints are added as penalization terms by
#
# :raw-latex:`\begin{equation}\label{Ineq_EF}
#  \lambda_1  \left(\sum_{i=1}^n l_i x_i - \sum_{k=0}^{N-1} 2^k s_k - B\right)^2. \tag{8}
#  \end{equation}`
#
# Combining Eq.(:raw-latex:`\ref{QUBO_form}`) and the two kinds of
# constraints Eq.(:raw-latex:`\ref{EQ_F}`) and
# Eq.(:raw-latex:`\ref{Ineq_EF}`), the general QUBO representation of a
# given combinatorial optimization problem is given by
#
# :raw-latex:`\begin{equation}\label{QUBO}
#  \min_x \left(\sum_{i=1}^{n-1} \sum_{j > i}^n c_{ij}x_{i}x_{j} + \sum_{i=1}^n h_i x_i + \lambda_0  \left(\sum_{i=1}^n c_i x_i - C\right)^2
# +  \lambda_1  \left(\sum_{i=1}^n l_i x_i - \sum_{k=0}^{N-1} 2^k s_k - B\right)^2\right). \tag{10}
#  \end{equation}`
#
# Remember that
#
# :raw-latex:`\begin{equation}\label{QE}
# \left(\sum_{i=0}^{n} c_i x_i - C\right)^2 = 2\sum_{i}^{n-1}\sum_{j>i}^{n} c_i c_j x_i x_j + \sum_{i}^{n} c_i^2 x_i - 2 C \sum_{i}^{n} c_i x_i + C^2\tag{11}
# \end{equation}`
#
# Following the same principle, more constraints can be added and note
# that after some manipulations, Eq.(:raw-latex:`\ref{QUBO}`) can be
# rewritten in the form of Eq.(:raw-latex:`\ref{QUBO_form}`) using the
# expansion in Eq. (:raw-latex:`\ref{QE}`). The last step to represent the
# QUBO problem on QPUs is to change the :math:`x_i` variables to spin
# variables :math:`z_i \in \{1, -1\}` by the transformation
# :math:`x_i = (1 - z_i) / 2`. Hence, Eq.(:raw-latex:`\ref{QUBO_form}`)
# can be represented by an Ising Hamiltonian with quadratic and linear
# terms plus a constant :math:`O`.
#
# :raw-latex:`\begin{equation}\label{IsingH}
# H_c(\mathrm{z}) = \sum_{i, j > i}^{n} J_{ij} z_i z_j + \sum_{i=1}^n h_{i}z_i + O\tag{12}.
# \end{equation}`
#
# Here, :math:`J_{ij}` are interaction terms and :math:`h_i` are linear
# terms, all of them depending on the combinatorial optimization problem.
#

from sympy import Symbol
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pennylane as qml

label_size = 16
plt.rcParams["xtick.labelsize"] = label_size
plt.rcParams["ytick.labelsize"] = label_size
plt.rcParams["axes.labelsize"] = label_size
plt.rcParams["legend.fontsize"] = label_size
# %matplotlib inline


######################################################################
#  2. The Knapsack Problem
# ------------------------
#
# In the knapsack problem (KP), a set of items with associated weights and
# values should be stored in a knapsack. The problem is to maximize the
# value of the items transported in the knapsack. The KP is restricted by
# the maximum weight the knapsack can carry. The KP is the simplest
# nontrivial integer programming model with binary variables, only one
# constraint, and positive coefficients. It is formally defined by
#
# :raw-latex:`\begin{equation}\label{Eq.1}
# \max  \sum_{i=1}^{n} p_{i} x_{i},
# \end{equation}`
#
# :raw-latex:`\begin{equation}\label{Eq.2}
# \sum_{i=1}^{n} w_{i} x_{i} \leq W,
# \end{equation}`
#
# where :math:`n` is the number of items, :math:`p_{i}` and :math:`w_{i}`
# are the value and weight of the :math:`ith` item, respectively,
# :math:`x_i` is the binary variable that represents whether the
# :math:`ith` item is in the knapsack or not, and W is the maximum weight
# that the knapsack can transport.
#
# In this tutorial, we will generate an instance of the `knapsack
# problem <https://en.wikipedia.org/wiki/Knapsack_problem>`__ and solve it
# using the quantum approximate optimization algorithm
# (`QAOA <https://arxiv.org/abs/1411.4028>`__). Our goal is to understand
# the different steps to encode a combinatorial optimization problem as an
# Ising Hamiltonian, how the QAOA function works, and how postprocessing
# the results of QAOA.
#
#  2.1 Example
# ~~~~~~~~~~~~
#
# A knapsack problem with 3 items with weights :math:`w_i = [1, 2, 3]`,
# values :math:`v_i=[5, 2, 4]`, and the knapsack maximum weight
# :math:`W_{max}=3`,
#
# :raw-latex:`\begin{equation}\label{QUBO_form}
# f(\mathrm{x}) = \sum_{i=1}^{3} v_{i}x_{i} \tag{12}
# \end{equation}`
#
# and inequality constraints given by
#
# :raw-latex:`\begin{equation}\label{inequality}
# W_{max} - \sum_{i=1}^3 w_i x_i \ge 0, \tag{14}
# \end{equation}`
#
# The problem has a QUBO formulation given by Eq.
# :raw-latex:`\ref{QUBO_K}` :raw-latex:`\begin{equation}\label{QUBO_K}
# \min_x \sum_{i=1}^n -v_i x_i + \lambda_1  \left( W_{max} - \sum_{i=1}^{3}w_i x_i -\sum_{k=0}^{N-1} 2^k s_k \right)^2, \tag{15}
# \end{equation}`
#
# where
# :math:`N = \lceil \log_2(\max_x W_{max} - \sum_{i=1}^n w_i x_i)\rceil = \log_2(W_{max})`.
#


def Knapsack(values: list, weights: list, max_weight: int, penalty: float):
    n_items = len(values)  # number of variables
    n_slacks = int(np.ceil(np.log2(max_weight)))  # number of slack variables

    x = {i: Symbol(f"x{i}") for i in range(n_items)}  # variables that represent the items
    S = sum(
        2**k * Symbol(f"s{k}") for k in range(n_slacks)
    )  # the slack variable in binary representation

    # objective function --------
    cost_fun = -sum(
        [values[i] * x[i] for i in x]
    )  # maximize the value of the items trasported Eq.12
    # (Note that minimizing the negative of cost function is the same that maximizing it)

    # ---------    constraint   Eq. 14  ----------
    constraint = max_weight - sum(weights[i] * x[i] for i in x) - S  # inequality constraint

    cost = (
        cost_fun + penalty * constraint**2
    )  # Eq. 15 cost function with penalization term for the Knapsack problem
    return cost


values = [5, 2, 4]
weights = [1, 2, 3]
max_weight = 3
penalty = 2  # lambda_1
qubo = Knapsack(values, weights, max_weight, penalty)  # Eq. 10 QUBO formulation
print(r"QUBO: min_x", qubo)


######################################################################
# Note that :math:`x_{i} x_{i} \equiv x_{i}`\ (if :math:`x_i = 0`,
# :math:`x_i^2 =0` and :math:`x_i=1`, :math:`x_i^2 = 1`), therefore
#

# Expanding and replacing the quadratic terms xi*xi = xi
qubo = qubo.expand().subs({symbol**2: symbol for symbol in qubo.free_symbols})
qubo


######################################################################
#   2.1 Ising Hamiltonian
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The last step to represent the QUBO problem on QPUs is to change the
# :math:`x_i \in \{0, 1\}` variables to spin variables
# :math:`z_i \in \{1, -1\}` by the transformation
# :math:`x_i = (1 - z_i) / 2` (Eq.11).
#

new_vars = {xi: (1 - Symbol(f"z{i}")) / 2 for i, xi in enumerate(qubo.free_symbols)}
new_vars

ising_Hamiltonian = qubo.subs(new_vars)
ising_Hamiltonian = ising_Hamiltonian.expand().simplify()
print("H(z) =", ising_Hamiltonian)


######################################################################
#   2.2 Brute force solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The first option to solve the knapsack problem is to use the brute force
# method. This method evaluates all the possible solutions of the QUBO and
# returns the one with the minimum cost. However, the number of possible
# solutions scales as :math:`2^n` where :math:`n` is the number of items
# in the problem, this makes this solution unfeasible for large instances.
#


def brute_force(qubo):
    vars_ = qubo.free_symbols
    n_vars = len(vars_)
    cost = {}
    min_cost = (0,)
    for i in range(2**n_vars):
        string = np.binary_repr(i, n_vars)
        cost[string] = qubo.subs({var: s for var, s in zip(vars_, string)})
        if cost[string] < min_cost[0]:
            min_cost = (cost[string], string)
    return cost, min_cost


sol_brute = brute_force(qubo)
optimal = {var: int(s) for var, s in zip(qubo.free_symbols, sol_brute[1][1])}
sol_str = sol_brute[1][1]
print(f"Optimal result: {optimal} | cost:{sol_brute[1][0]}")


######################################################################
#  3. QAOA Circuit
# ----------------
#
# Finally, we use `QAOA <https://arxiv.org/pdf/1411.4028.pdf>`__ to find
# the solution to our Knapsack problem. In this case, the cost
# Hamiltonian, :math:`H(z)`, obtained from the QUBO formulation, is
# translated into a parametric unitary gate given by
#
# :raw-latex:`\begin{equation}\label{UC}
#     U(H_c, \gamma)=e^{-i \gamma H_c},\tag{16}
# \end{equation}` where :math:`\gamma` is a parameter to be optimized. A
# second unitary operator applied is
#
# :raw-latex:`\begin{equation}\label{UB}
#     U(B, \beta)=e^{i \beta X},\tag{17}
# \end{equation}`
#
# where :math:`\beta` is the second parameter that must be optimized and
# :math:`X = \sum_{i=1}^n \sigma_i^x` with :math:`\sigma_i^x` the Pauli-x
# quantum gate applied to qubit :math:`i`. The general QAOA circuit is
# shown in **Fig.1**. Here,
# :math:`R_X(\theta) = e^{-i \frac{\theta}{2} \sigma_x}`, :math:`p`
# represents the number of repetitions of the unitary gates
# Eqs.:raw-latex:`\ref{UC}` and :raw-latex:`\ref{UB}` with each repetition
# having separate values for :math:`\gamma_p` and :math:`\beta_p`, and the
# initial state is a superposition state :math:`| + \rangle^{\otimes n}`.
#
# .. raw:: html
#
#    <center>
#
# \ **Fig.1** Schematic representation of QAOA for :math:`p` layers. The
# parameters :math:`\gamma` and :math:`\beta` for each layer are the ones
# to be optimized.
#
# .. raw:: html
#
#    </center>
#

num_qubits = len(ising_Hamiltonian.free_symbols)
dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev)
def qaoa_circuit(gammas, betas, ising_Hamiltonian):
    p = len(gammas)
    ising_dict = ising_Hamiltonian.as_coefficients_dict()
    norm = float(max(ising_dict.values()))
    # Apply the initial layer of Hadamard gates to all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    # repeat p layers the circuit shown in Fig. 1
    for l in range(p):
        for i in range(num_qubits - 1):  # single-qubit terms
            if Symbol(f"z{i}") in ising_dict:
                wi = float(ising_dict[Symbol(f"z{i}")])
                qml.RZ(2 * gammas[l] * wi / norm, wires=i)
            for j in range(i + 1, num_qubits):  # two-qubit terms
                if Symbol(f"z{i}") * Symbol(f"z{j}") in ising_dict:
                    wij = float(ising_dict[Symbol(f"z{i}") * Symbol(f"z{j}")])
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gammas[l] * wij / norm, wires=j)
                    qml.CNOT(wires=[i, j])

        for i in range(num_qubits):
            qml.RX(2 * betas[l], wires=i)
    return qml.probs()


qaoa_circuit([0.5], [0.5], ising_Hamiltonian)


######################################################################
#   3.2 Optimization
# ~~~~~~~~~~~~~~~~~~
#
# Once we define the QAOA circuit of the combinatorial optimization
# problem, the next step is to find values of :math:`\beta_0` and
# :math:`\gamma_0` that minimize the expectation value of the Ising
# Hamiltonian. Here, we use ``pennylane`` and ``sympy`` to find the cost
# function minimum value. In this case, we use the ``Powell`` optimization
# method with a maximum iteration equal to 100.
#

callback_f = {"fx": [], "params": []}


def cost_function(parameters, objective):
    """
    Return a cost function that depends of the QAOA circuit

    Parameters
    ----------
    parameters : list
        gamma and beta values of the QAOA circuit. [gamma_0, gamma_1,..., gamma_p-1, beta_0, beta_1, ..., beta_p-1]
    objective : sympy Ising Hamiltonian formulation
        Objective function of the problem

    Returns
    -------
    float
        Cost of the evaluation of n string on the objective function

    """
    p = len(parameters) // 2
    gammas = parameters[:p]
    betas = parameters[p:]
    # running the QAOA circuit using pennylane
    probs = qaoa_circuit(gammas, betas, objective)
    cost = np.zeros((len(probs),))
    # The pennylane result is a list with probabilities in order
    for i, p in enumerate(probs):
        sample = np.binary_repr(i, len(objective.free_symbols))
        dict_sol = {f"z{ni}": 1 - 2 * int(bi) for ni, bi in enumerate(sample)}
        feval = objective.subs(dict_sol)  # evaluate the QUBO
        cost[i] = p * float(feval)
    callback_f["fx"].append(cost.mean())
    callback_f["params"].append(parameters)
    return cost.mean()


seed = 123
np.random.seed(seed)
x0 = [-0.5, -0.1]  # Initial guessing
# -------- Simpy minimization method to find optimal parameters for beta and gamma
sol = minimize(
    cost_function, x0=x0, args=(ising_Hamiltonian), method="Powell", options={"maxiter": 100}
)
sol


######################################################################
#   3.3 Results visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Optimization steps (Ising Hamiltonian expectation value)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# expectation value vs. iterations
fig, ax = plt.subplots()
ax.plot(callback_f["fx"])
ax.set_xlabel("iterations")
ax.set_ylabel(r"$\langle H(z)\rangle$")
ax.grid()
ax.set_title("QAOA")


######################################################################
# Probability distribution visualization
# --------------------------------------
#
# Use QAOA to improve the probability of getting the optimal solutions.
# Note that not all of the solutions are valid and remember that QAOA in
# general does not find the optimal solution but a probability
# distribution where optimal and suboptimal solutions are more probable
# generally.
#

probabilites = qaoa_circuit(
    [sol.x[0]], [sol.x[1]], ising_Hamiltonian
)  # Run the QAOA circuit using the betas and gammas found
results = {
    np.binary_repr(i, len(ising_Hamiltonian.free_symbols)): p for i, p in enumerate(probabilites)
}
opt_res = {sol_str: results[sol_str]}  # probability of the optimal solution
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar([int(k, 2) for k in results.keys()], results.values())
ax.bar(
    [int(k, 2) for k in results.keys() if k in opt_res],
    [v for k, v in results.items() if k in opt_res],
    color="tab:red",
    label="optimal",
)
ax.set_xticks(range(2 ** len(qubo.free_symbols)))
ticks = ax.set_xticklabels(
    [np.binary_repr(i, len(qubo.free_symbols)) for i in range(2 ** len(qubo.free_symbols))],
    rotation=90,
)
ax.set_ylabel("Count", fontsize=18)
ax.legend()
ax.set_title("QAOA with optimized parameters solutions count", fontsize=18)


######################################################################
# ``qml.probs()`` gives the exact probability distribution of the QAOA
# algorithm for the optimal solution.
#

print(
    f"Probability of finding the optimal solution using QAOA: {100*np.round(results[sol_str],3)}%"
)
print(f"Random guessing: {100/2**len(sol_str)}%")


######################################################################
# Landscape
# ---------
#
# For the case where there is just one layer on the QAOA, we can visualize
# the energy expectation value :math:`\langle H(z) \rangle` for the
# knapsack problem. The Figure below shows the landscape for the Knapsack
# problem with the optimal solution of the optimization step.
#
# Let’s start by iterating over different values of :math:`\beta` and
# :math:`\gamma`:
#

n1 = n2 = 50
gammas = np.linspace(-np.pi / 2, np.pi / 2, n1)
betas = np.linspace(-np.pi / 2, np.pi / 2, n2)

landscape = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        landscape[i, j] = cost_function([gammas[i], betas[j]], ising_Hamiltonian)

fig, ax = plt.subplots(figsize=(5, 5))
ax1 = ax.imshow(landscape, cmap="coolwarm", extent=[-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2])
ax.plot(
    sol.x[1],
    sol.x[0],
    marker="*",
    markersize=10,
    markeredgecolor="black",
    color="tab:red",
    label="optimal",
    linewidth=0,
)

ax.set_xticks([-np.pi / 2, 0, np.pi / 2])
ax.set_yticks([-np.pi / 2, 0, np.pi / 2])
ax.set_xticklabels([r"$-\pi/2$", 0, r"$\pi/2$"])
ax.set_yticklabels([r"$-\pi/2$", 0, r"$\pi/2$"])
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\gamma$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.27))
ax.set_title("Energy Landscape", fontsize=18)
plt.colorbar(ax1)


######################################################################
#  Exercise (Optional)
# --------------------
#
# In this excercise you must repeat the knapsack problem this time with
# some modifications. Now, we need to maximize the value transported on a
# knapsack with 6 items with weights :math:`w = [7, 2, 1, 3, 2, 5]`,
# values :math:`v = [4, 3, 2, 1, 5, 3]`, and maximum weight
# :math:`W_{max} = 15`. An additional restriction in this case is that
# just one of the items :math:`[x_1, x_3, x_5]` could be in the knapsack.
#
# :raw-latex:`\begin{equation}
# x_1 + x_3 + x_5 = 1
# \end{equation}`
#
# 1. Repeat the steps in `Example <#example>`__ and `QAOA <#qaoa>`__ for
#    this problem (note that you should modify the qubo formulation to
#    include the equality constraint.)
#


def knapsack_new(values, weights, max_weight):
    n_items = len(values)  # number of variables
    n_slacks = int(np.ceil(np.log2(max_weight)))  # number of slack variables

    x = {i: Symbol(f"x{i}") for i in range(n_items)}  # variables that represent the items
    S = sum(
        2**k * Symbol(f"s{k}") for k in range(n_slacks)
    )  # the slack variable in binary representation

    # objective function --------
    cost_fun = -sum(
        [values[i] * x[i] for i in x]
    )  # maximize the value of the items trasported Eq.12
    # (Note that minimizing the negative of cost function is the same that maximizing it)

    # ---------    constraint   Eq. 14  ----------
    constraint1 = max_weight - sum(weights[i] * x[i] for i in x) - S  # inequality constraint

    constraint2 = x[1] - 1

    cost = (
        cost_fun + 2 * constraint1**2 + 10 * constraint2**2
    )  # Eq. 15 cost function with penalization term for the Knapsack problem
    return cost

# About the author
# ----------------
#

