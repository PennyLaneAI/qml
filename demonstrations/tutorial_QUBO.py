"""
The Quadratic Unconstrained Binary Optimization (QUBO)
======================================================

*Author: Alejandro Montanez — Posted: XX May 2023.*

Solving combinatorial optimization problems using quantum computing is
one of those promising applications for the near term. But, why are
combinatorial optimization problems even important? well, we care about
them because we have useful applications that can be translated into
combinatorial optimization problems, in fields such as logistics,
finance, and engineering. But having useful applications is not enough,
and it is here where the second ingredient comes in, combinatorial
optimization problems are complex! and finding good solutions
(classically) for large instances of them requires an enormous amount of
computational resources 😮‍💨.

In this demo, we will be using a quantum algorithm called the Quantum
Approximate Optimization Algorithm (QAOA) to solve a combinatorial
optimization problem. QAOA is an algorithm that leverages the power of
quantum computers to find the ground state of a target Hamiltonian. We
will show throughout this demo how to encode the Knapsack as a target
Hamiltonian and solve it using QAOA.

"""


######################################################################
#  1. Generalities: Combinatorial Optimization Problems
# =====================================================
# 
# Combinatorial optimization problems are a type of mathematical problem
# that involves finding the best way to arrange a set of objects or values
# to achieve a specific goal. The word ‘combinatorial’ refers to the fact
# that we are dealing with combinations of objects, while ‘optimization’
# refers to the fact that we are trying to find the best possible
# arrangement of them.
# 
# Let’s start with a basic example. Imagine we have 5 items ⚽️, 💻, 📸,
# 📚, and 🎸 and we would love to bring all of them with us. But to our
# bad luck, we only have a knapsack and do not have space for all of them
# 😔. So, we need to find the best way to bring the most important items
# for us.
# 
# But, to start the formulation of our problem we need a little more
# information. From our problem statement, we know that we need to
# maximize the value of the most important items. So we need to assign a
# value based on the importance the items have to us:
# 

items_values = {"⚽️":3, "💻":3, "📸":1, "📚":1, "🎸":5 }


######################################################################
# Additionally, we know that we are restricted for the space in the
# Knapsack, for simplification let’s assume the restriction is in terms of
# weight. So we need to assign an estimate of the weight of each item:
# 

items_weight = {"⚽️":2, "💻":4, "📸":1, "📚":3, "🎸":5 }


######################################################################
# Finally, we need to know what is the maximum weight the knapsack can
# bring:
# 

max_weight = 7


######################################################################
# Ok, now that we have a problem to work with. Let’s start with the
# easiest way to solve them, it is trying all possible combinations of
# those items which we find to grow as :math:`2^n` where :math:`n` is the
# number of items. Why does it grow in that way? because for each item we
# have two options “1” if we bring the item and “0” otherwise. So 2
# options for each item, and we have 5 items then
# :math:`2*2*2*2*2 = 2^5 = 32` combinations in our case. For each of these
# cases, we calculate the overall value and weight of the items carried,
# and from them, we selected the one that fulfills the weight constraint
# and has the largest value (the optimization step).
# 

import numpy as np
import pandas as pd
items = list(items_values.keys())
n = len(items)
combinations = {}
for case_i in range(2**n):
    bitstring = np.binary_repr(case_i, n) #bitstring representation of a case, e.g, "01100" in our problem means bringing (-💻📸--)
    combinations[case_i] = {"items":[], "value":0, "weight":0}
    max_value = 0 
    for item_i, bring in enumerate(bitstring):
         if bring == "1":
            combinations[case_i]["items"].append(items[item_i])
            combinations[case_i]["value"] += items_values[items[item_i]]
            combinations[case_i]["weight"] += items_weight[items[item_i]]
    if combinations[case_i]["value"] > max_value and combinations[case_i]["weight"] <= max_weight:
        max_value = combinations[case_i]["value"]
        optimal_solution = {"items": combinations[case_i]["items"],
                           "value": combinations[case_i]["value"],
                           "weight":combinations[case_i]["weight"]}
pd.DataFrame(combinations)

print(f"The best combination is {optimal_solution['items']} with a total value: {optimal_solution['value']} and total weight {optimal_solution['weight']} ")


######################################################################
# That was easy, right? But what about if we have larger cases like 10,
# 50, or 100? Just to play around with these numbers suppose we spend 1 ns
# to try one case.
# 

print(f"- For 10 items, 2^10 cases, we need {2**10*1e-9} seconds to solve the problem\n- For 50 items, 2^50 cases, we need:{round((2**50*1e-9)/(3600*24))} days \n- For 100 items, 2^100 cases, we need: {round((2**100*1e-9)/(3600*24*365))} years")


######################################################################
# I guess we don’t have the time to try all possible solutions for 100
# items 😅! Thankfully, we don’t need to try all of them and there are
# algorithms to find good solutions to combinatorial optimization
# problems, and maybe one day we will show that one of these algorithms is
# quantum. So let’s continue with our quest 🫡.
# 
# Our next step is to represent our problem mathematically. Well, we know
# what we want, **maximize** the value of the items transported, so let’s
# create a function :math:`f(\mathrm{x})` with these characteristics. To
# do so, assign to the items, the variables :math:`x_i` for each of them
# :math:`\mathrm{x} = \{x_0:"⚽️", x_1:"💻", x_2:"📸", x_3:"📚", x_4:"🎸"\}`
# and multiply such variable for the value of the item, items_value =
# {“⚽️”:3, “💻”:3, “📸”:1, “📚”:1, “🎸”:5 }.
# 
# .. math:: \max_x f(\mathrm{x}) = 3x_0 + 3x_1 + x_2 + x_3 + 5x_4 \tag{1}
# 
# This function clearly represents the value of the items we transport and
# in mathematical optimization usually, it is called the
# ``objective function``. Usually, solvers do not optimize to maximize a
# function, instead, they do for minimizing it, so a simple trick in our
# case is to minimize the negative of our function (which ends up
# maximizing our original function 🤪)
# 
# .. math:: \min_x f(\mathrm{x}) = -(3x_0 + 3x_1 + x_2 + x_3 + 5x_4) \tag{2}
# 

from sympy import Symbol

x = {items[i]: Symbol(f"x{i}") for i in range(n)}  # variables that represent the items
fx = 0
for i in range(n):
    fx -= items_values[items[i]] * x[items[i]]
print("f(x) =", fx)


######################################################################
# But just with this function, we cannot solve the problem. We also need
# weight restriction. Based on our variables, the weight list
# (items_weight = {“⚽️”:2, “💻”:4, “📸”:1, “📚”:3, “🎸”:5 }), and the
# knapsack maximum weight (max_weight = 7), we can construct our
# restriction
# 
# .. math:: 2x_0 + 4x_1 + x_2 + x_3 + 5x_4 \le 7 \tag{3}
# 
# Now, here is an important part of our model, we need to find a way to
# combine our ``objective function`` with this ``inequality constraint``.
# One common method is to include the constraint as a **penalization**
# term in the objective function. This penalization term should be zero
# when the total weight of the items is less or equal to 7 and large
# otherwise. So to make them zero in the range of validity of the
# constraint, the usual approach is to use ``slack variables`` (an
# alternative method `here <https://arxiv.org/pdf/2211.13914.pdf>`__ 😉).
# 
# The slack variable is an auxiliary variable to convert inequality
# constraints into equality constraints. The slack variable :math:`S`
# represents the amount by which the left-hand side of the inequality
# falls short of the right-hand side. If the left-hand side is less than
# the right-hand side, then :math:`S` will be positive and equal to the
# difference between the two sides. In our case
# 
# .. math:: 2x_0 + 4x_1 + x_2 + x_3 + 5x_4 + S = 7 \tag{4}
# 
# for :math:`0 \le S \le 7`. But let’s take this slowly because we can get
# lost here, so let’s see this with some examples…
# 
# -  Imagine this case, no item is selected {:math:`x_0`:0, :math:`x_1`:0,
#    :math:`x_2`:0, :math:`x_3`:0, :math:`x_4`:0}, so the overall weight
#    is zero (a valid solution) and the equality constraint Eq.(4) must be
#    fulfilled. So we select our slack variable to be 7.
# 
# -  Now, what if we bring the ⚽️, 📸, 📚 {:math:`x_0`:1, :math:`x_1`:0,
#    :math:`x_2`:1, :math:`x_3`:1, :math:`x_4`:0} so the overall weight is
#    2 + 1 + 1=4 (a valid solution) and to make the equality constraint
#    right S=3.
# 
# -  Finally, what if we try to bring all the items {:math:`x_0`:1,
#    :math:`x_1`:1, :math:`x_2`:1, :math:`x_3`:1, :math:`x_4`:1}, the
#    total weight, in this case, is 2+4+1+3+5=15 (not a valid solution),
#    to fulfill the constraint, we need :math:`S = -7` but the slack
#    variable is in the range :math:`[0,7]` in our definition, so, in this
#    case, there is no way to represent the right-hand side in our
#    equation.
# 
# Excellent, now we have a way to represent the inequality constraint. Two
# further steps are needed, first, the slack variable has to be
# represented in binary form so we can transform it by
# 
# .. math:: S = \sum_{k=0}^{N-1} 2^k s_k
# 
# where :math:`N = \lceil\log_2(\max S + 1)\rceil`. In our case
# :math:`N = \lceil\log_2(7 + 1) = 3\rceil`. We need three binary
# variables to represent the range of our :math:`S` variable.
# 
# .. math:: S = 2^0 s_0 + 2^1 s_1 + 2^2 s_2 = s_0 + 2s_1 + 4s_2 
# 
# For example, if we need to represent the second case above (⚽️, 📸, 📚)
# :math:`S=3\rightarrow\{s_0:1, s_1:1,s_2:0\}`.
# 
# We are almost done in our quest to represent our problem in such a way
# that our quantum computer can manage it. The last step is to add the
# penalization term, a usual choice for it is to use a quadratic
# penalization
# 
# .. math:: p(x,s) = \lambda \left(2x_0 + 4x_1 + x_2 + x_3 + 5x_4 + s_0 + 2 s_1 + 4s_2 - 7\right)^2 \tag{5}
# 
# note that this is the same Eq.(4) just the left-hand side less the
# right-hand side here. With this expression just when the condition is
# satisfied the term inside the parenthesis is zero. :math:`\lambda` is a
# penalization coefficient that we must tune to make that the constraint
# will be always fulfilled.
# 

N = round(np.ceil(np.log2(max_weight)))
slack = {f"s{k}":Symbol(f"s{k}") for k in range(N)}
Lambda = Symbol(r"\lambda")
p = 0
for i in range(n):
    p += items_weight[items[i]] * x[items[i]]
for k in range(N):
    p += 2**k * slack[f"s{k}"]

p = Lambda * p**2
print("p(x,s) =", p)


######################################################################
# Then we have all the ingredients to construct our function, combining
# Eq.(2) and Eq.(5):
# 
# .. math:: \min_{x,s} f(\mathrm{x}) = -(3x_0 + 3x_1 + x_2 + x_3 + 5x_4) + \lambda \left(2x_0 + 4x_1 + x_2 + x_3 + 5x_4 + s_0 + 2 s_1 + 4s_2 - 7\right)^2 \tag{6}
# 
# Let’s see this penalization in action, recall our third example trying
# to bring all the items (“⚽️”, “💻”, “📸”, “📚”, “🎸”) Eq.(6) gives
# 
# .. math:: \min_{s} -\color{blue}{(3(1) + 3(1) + (1) + (1) + 5(1))} + \lambda \color{green}{\left(2(1) + 4(1) + (1) + (1) + 5(1) + s_0 + 2 s_1 + 4s_2 - 7\right)^2} = - 13 + \lambda(-7)^2 = -13 + \lambda49
# 
# Note the to minimize the constraint :math:`s_0, s_1, s_2` must be zero.
# 


######################################################################
# <——- Next steps from here
# =========================
# 


######################################################################
# These problems often involve making choices and trade-offs between
# different options, and finding the solution usually requires a
# systematic approach that considers all possible arrangements. For
# example, one common combinatorial optimization problem is the knapsack
# problem, which involves packing a set of items into a knapsack with
# limited capacity while maximizing the total value of the items packed.
# 
# Let’s go step by step in the details and put it mathematically! First,
# let’s start by the word best, in this case, the objective function will
# indicate which solution is the best. Usually, we can describe the
# general form of a combinatorial optimization problem solvable by a
# quantum device by
# 
# .. math::
# 
# 
#    f(\mathrm{x}) = \sum_{i=1}^{n-1} \sum_{j > i}^n q_{ij}x_{i}x_{j} + \frac{1}{2}\sum_{i=1}^n q_{ii} x_i,\tag{2}
# 
# where :math:`n` is the number of variables,
# :math:`q_{ij} \in \mathbb{R}` are coefficients associated to the
# specific problem, and :math:`x_i \in \{0,1\}` are the binary variables
# of the problem. Note that :math:`x_{i} x_{i} \equiv x_{i}` and
# :math:`q_{ij} = q_{ji}` in this formulation.
# 
# The set of combinatorial problems that can be represented by the QUBO
# formulation is characterized by functions of the form
# 
# .. math::
# 
# 
#    f(\mathrm{x}) = \frac{1}{2}\sum_{i=1}^{n} \sum_{j=1}^n q_{ij} x_{i} x_{j}, \tag{1}
# 
# Therefore, and equality constraints are given by
# 
# .. math::
# 
# 
#    \sum_{i=1}^n c_i x_i = C, \ c_i \in \mathbb{Z}, \tag{3}
# 
# and inequality constraints are given by
# 
# .. math::
# 
# 
#    \sum_{i=1}^n l_i x_i \le B, \ l_i \in \mathbb{Z} \tag{4}
# 
# where :math:`C` and :math:`B` are constants. To transform these problems
# into the QUBO formulation the constraints are added as penalization
# terms. In this respect, the equality constraints are included in the
# cost function using the following penalization term
# 
# .. math::
# 
# 
#    \lambda_0 \left(\sum_{i=1}^n c_i x_i - C\right)^2,\tag{5}
# 
# where :math:`\lambda_0` is a penalization coefficient that should be
# chosen to guarantee that the equality constraint is fulfilled. In the
# case of inequality constraint, the common approach is to use a `slack
# variable <https://en.wikipedia.org/wiki/Slack_variable>`__. The slack
# variable, :math:`S`, is an auxiliary variable that makes a penalization
# term vanish when the inequality constraint is achieved,
# 
# .. math::
# 
# 
#     B -\sum_{i=1}^n l_i x_i - S = 0.\tag{6}
# 
# Therefore, when Eq.(4) is satisfied, Eq.(6) is already zero. This means
# the slack variable, :math:`S`, must be in the range
# :math:`0 \le S \le \max_x \sum_{i=1}^n B - l_i x_i`. To represent the
# :math:`slack` variable in binary form, the slack is decomposed in binary
# variables:
# 
# .. math::
# 
# 
#    S = \sum_{k=0}^{N-1} 2^k s_k,\tag{7}
# 
# where :math:`s_k` are the slack binary variables. Then, the inequality
# constraints are added as penalization terms by
# 
# .. math::
# 
# 
#     \lambda_1  \left(\sum_{i=1}^n l_i x_i - \sum_{k=0}^{N-1} 2^k s_k - B\right)^2. \tag{8}
# 
# Combining Eq.(2) and the two kinds of constraints Eq.(3) and Eq.(4), the
# general QUBO representation of a given combinatorial optimization
# problem is given by
# 
# .. math::
# 
# 
#     \min_x \left(\sum_{i=1}^{n-1} \sum_{j > i}^n c_{ij}x_{i}x_{j} + \sum_{i=1}^n h_i x_i + \lambda_0  \left(\sum_{i=1}^n c_i x_i - C\right)^2
#    +  \lambda_1  \left(\sum_{i=1}^n l_i x_i - \sum_{k=0}^{N-1} 2^k s_k - B\right)^2\right). \tag{10}
# 
# Remember that
# 
# .. math::
# 
# 
#    \left(\sum_{i=0}^{n} c_i x_i - C\right)^2 = 2\sum_{i}^{n-1}\sum_{j>i}^{n} c_i c_j x_i x_j + \sum_{i}^{n} c_i^2 x_i - 2 C \sum_{i}^{n} c_i x_i + C^2\tag{11}
# 
# Following the same principle, more constraints can be added and note
# that after some manipulations, Eq.(10) can be rewritten in the form of
# Eq.(2) using the expansion in Eq. (11). The last step to represent the
# QUBO problem on QPUs is to change the :math:`x_i` variables to spin
# variables :math:`z_i \in \{1, -1\}` by the transformation
# :math:`x_i = (1 - z_i) / 2`. Hence, Eq.(10) can be represented by an
# Ising Hamiltonian with quadratic and linear terms plus a constant
# :math:`O`.
# 
# .. math::
# 
# 
#    H_c(\mathrm{z}) = \sum_{i, j > i}^{n} J_{ij} z_i z_j + \sum_{i=1}^n h_{i}z_i + O\tag{12}.
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
plt.rcParams['xtick.labelsize'] = label_size 
plt.rcParams['ytick.labelsize'] = label_size 
plt.rcParams['axes.labelsize'] = label_size 
plt.rcParams['legend.fontsize'] = label_size 
# %matplotlib inline


######################################################################
#  2. The Knapsack Problem
# ========================
# 
# In the knapsack problem (KP), a set of items with associated weights and
# values should be stored in a knapsack. The problem is to maximize the
# value of the items transported in the knapsack. The KP is restricted by
# the maximum weight the knapsack can carry. The KP is the simplest
# nontrivial integer programming model with binary variables, only one
# constraint, and positive coefficients. It is formally defined by
# 
# .. math::
# 
# 
#    \max  \sum_{i=1}^{n} p_{i} x_{i},
# 
# .. math::
# 
# 
#    \sum_{i=1}^{n} w_{i} x_{i} \leq W, 
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


######################################################################
#  2.1 Example
# ------------
# 
# A knapsack problem with 3 items with weights :math:`w_i = [1, 2, 3]`,
# values :math:`v_i=[5, 2, 4]`, and the knapsack maximum weight
# :math:`W_{max}=3`,
# 
# .. math::
# 
# 
#    f(\mathrm{x}) = \sum_{i=1}^{3} v_{i}x_{i} \tag{12}
# 
# and inequality constraints given by
# 
# .. math::
# 
# 
#    W_{max} - \sum_{i=1}^3 w_i x_i \ge 0, \tag{13}
# 
# The problem has a QUBO formulation given by
# 
# .. math::
# 
# 
#    \min_x \sum_{i=1}^n -v_i x_i + \lambda_1  \left( W_{max} - \sum_{i=1}^{3}w_i x_i -\sum_{k=0}^{N-1} 2^k s_k \right)^2, \tag{14}
# 
# where
# :math:`N = \lceil \log_2(\max_x W_{max} - \sum_{i=1}^n w_i x_i)\rceil = \log_2(W_{max})`.
# 

def Knapsack(values: list, weights: list, max_weight: int, penalty:float):
    n_items = len(values) # number of variables
    n_slacks = int(np.ceil(np.log2(max_weight))) # number of slack variables
    
    x = {i: Symbol(f"x{i}") for i in range(n_items)}  # variables that represent the items
    S = sum(2**k * Symbol(f"s{k}") for k in range(n_slacks)) # the slack variable in binary representation
    
    # objective function --------
    cost_fun = - sum([values[i]*x[i] for i in x]) # maximize the value of the items trasported Eq.12
    #(Note that minimizing the negative of cost function is the same that maximizing it)
    
    # ---------    constraint   Eq. 14  ----------
    constraint = max_weight - sum(weights[i] * x[i] for i in x) - S #inequality constraint

    cost = cost_fun + penalty * constraint**2 # Eq. 15 cost function with penalization term for the Knapsack problem
    return cost

values = [5, 2, 4]
weights = [1, 2, 3]
max_weight = 3
penalty = 2 #lambda_1
qubo = Knapsack(values, weights, max_weight, penalty) # Eq. 10 QUBO formulation
print(r'QUBO: min_x', qubo)


######################################################################
# Note that :math:`x_{i} x_{i} \equiv x_{i}`\ (if :math:`x_i = 0`,
# :math:`x_i^2 =0` and :math:`x_i=1`, :math:`x_i^2 = 1`), therefore
# 

# Expanding and replacing the quadratic terms xi*xi = xi
qubo = qubo.expand().subs({symbol**2:symbol for symbol in qubo.free_symbols})
qubo


######################################################################
#   2.2 Ising Hamiltonian
# -----------------------
# 
# The last step to represent the QUBO problem on QPUs is to change the
# :math:`x_i \in \{0, 1\}` variables to spin variables
# :math:`z_i \in \{1, -1\}` by the transformation
# :math:`x_i = (1 - z_i) / 2`.
# 

new_vars = {xi:(1 - Symbol(f"z{i}"))/2 for i, xi in enumerate(qubo.free_symbols)}
new_vars

ising_Hamiltonian = qubo.subs(new_vars)
ising_Hamiltonian = ising_Hamiltonian.expand().simplify()
print("H(z) =", ising_Hamiltonian)


######################################################################
#   2.3 Brute force solution
# --------------------------
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
    min_cost = (0, )
    for i in range(2**n_vars):
        string = np.binary_repr(i, n_vars)
        cost[string] = qubo.subs({var:s for var, s in zip(vars_, string)})
        if cost[string] < min_cost[0]:
            min_cost = (cost[string], string)
    return cost, min_cost
sol_brute = brute_force(qubo)
optimal = {var:int(s) for var, s in zip(qubo.free_symbols, sol_brute[1][1])}
sol_str = sol_brute[1][1]
print(f"Optimal result: {optimal} | cost:{sol_brute[1][0]}")


######################################################################
#  3. QAOA
# --------
# 
# Finally, we use `QAOA <https://arxiv.org/pdf/1411.4028.pdf>`__ to find
# the solution to our Knapsack problem. In this case, the cost
# Hamiltonian, :math:`H(z)`, obtained from the QUBO formulation, is
# translated into a parametric unitary gate given by
# 
# .. math::
# 
# 
#        U(H_c, \gamma)=e^{-i \gamma H_c},\tag{16}
# 
# where :math:`\gamma` is a parameter to be optimized. A second unitary
# operator applied is
# 
# .. math::
# 
# 
#        U(B, \beta)=e^{i \beta X},\tag{17}
# 
# where :math:`\beta` is the second parameter that must be optimized and
# :math:`X = \sum_{i=1}^n \sigma_i^x` with :math:`\sigma_i^x` the Pauli-x
# quantum gate applied to qubit :math:`i`. The general QAOA circuit is
# shown in **Fig.1**. Here,
# :math:`R_X(\theta) = e^{-i \frac{\theta}{2} \sigma_x}`, :math:`p`
# represents the number of repetitions of the unitary gates Eqs.(16-17)
# with each repetition having separate values for :math:`\gamma_p` and
# :math:`\beta_p`, and the initial state is a superposition state
# :math:`| + \rangle^{\otimes n}`.
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
        for i in range(num_qubits-1): # single-qubit terms
            if Symbol(f"z{i}") in ising_dict:
                wi = float(ising_dict[Symbol(f"z{i}")])
                qml.RZ(2 * gammas[l] * wi/norm, wires=i)
            for j in range(i+1, num_qubits): # two-qubit terms
                if Symbol(f"z{i}")*Symbol(f"z{j}") in ising_dict:
                    wij = float(ising_dict[Symbol(f"z{i}")*Symbol(f"z{j}")])
                    qml.CNOT(wires = [i, j])
                    qml.RZ(2 * gammas[l] * wij/norm, wires=j)
                    qml.CNOT(wires = [i, j])

        for i in range(num_qubits): 
            qml.RX(2 * betas[l], wires=i)
    return qml.probs()
qaoa_circuit([0.5],[0.5], ising_Hamiltonian)


######################################################################
#   3.1 Optimization
# ------------------
# 
# Once we define the QAOA circuit of the combinatorial optimization
# problem, the next step is to find values of :math:`\beta_0` and
# :math:`\gamma_0` that minimize the expectation value of the Ising
# Hamiltonian. Here, we use ``pennylane`` and ``sympy`` to find the cost
# function minimum value. In this case, we use the ``Powell`` optimization
# method with a maximum iteration equal to 100.
# 

callback_f = {"fx":[], "params":[]}
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
    p = len(parameters)//2
    gammas = parameters[:p]
    betas = parameters[p:]
    # running the QAOA circuit using pennylane
    probs = qaoa_circuit(gammas, betas, objective)
    cost = np.zeros((len(probs),))
    # The pennylane result is a list with probabilities in order
    for i, p  in enumerate(probs):
        sample = np.binary_repr(i, len(objective.free_symbols))
        dict_sol = {f"z{ni}":1 - 2*int(bi) for ni, bi in enumerate(sample)}
        feval = objective.subs(dict_sol) #evaluate the QUBO
        cost[i] = p * float(feval)
    callback_f["fx"].append(cost.mean())
    callback_f["params"].append(parameters)
    return cost.mean()

seed = 123
np.random.seed(seed)
x0 = [-0.5, -0.1] # Initial guessing
#-------- Simpy minimization method to find optimal parameters for beta and gamma
sol = minimize(cost_function, x0 = x0, args=(ising_Hamiltonian), method="Powell", options={"maxiter":100})
sol


######################################################################
#   3.2 Results visualization
# ---------------------------
# 
# In this section, we will see different visualization strategies to
# interpret the results. First, the expectation vs. the number of
# iterations using the classical algorithm. Second, the probability
# distribution of the different bitstrings. Finally, the energy landscape
# for a specific region of the :math:`\beta` and :math:`\gamma`.
# 
# Expectation energy vs. iteration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Once, the optimization step has finished. We can visualize the result of
# the energy expectation value at each iteration. We have saved the
# information in ``callback_f`` for each of the steps of the ``Powell``
# algorithm.
# 

# expectation value vs. iterations
fig, ax = plt.subplots()
ax.plot(callback_f["fx"])
ax.set_xlabel("iterations")
ax.set_ylabel(r"$\langle H(z)\rangle$")
ax.grid()
ax.set_title("QAOA")


######################################################################
# Probability distribution
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We use QAOA to improve the probability of getting the optimal solutions.
# Note that not all of the solutions are valid and remember that QAOA in
# general does not find the optimal solution but a probability
# distribution where optimal and suboptimal solutions are more probable
# generally.
# 

probabilites = qaoa_circuit([sol.x[0]], [sol.x[1]], ising_Hamiltonian) #Run the QAOA circuit using the betas and gammas found
results = {np.binary_repr(i, len(ising_Hamiltonian.free_symbols)):p for i, p in enumerate(probabilites)}
opt_res = {sol_str:results[sol_str]} #probability of the optimal solution
fig, ax = plt.subplots(figsize=(20,5))
ax.bar([int(k, 2) for k in results.keys()], results.values())
ax.bar([int(k, 2) for k in results.keys() if k in opt_res], [v for k, v in results.items() if k in opt_res], color="tab:red", label="optimal")
ax.set_xticks(range(2**len(qubo.free_symbols)))
ticks = ax.set_xticklabels([np.binary_repr(i, len(qubo.free_symbols)) for i in range(2**len(qubo.free_symbols))], rotation=90)
ax.set_ylabel("Count", fontsize=18)
ax.legend()
ax.set_title("QAOA with optimized parameters solutions count", fontsize=18)


######################################################################
# ``qml.probs()`` gives the exact probability distribution of the QAOA
# algorithm for the optimal solution.
# 

print(f"Probability of finding the optimal solution using QAOA: {100*np.round(results[sol_str],3)}%")
print(f"Random guessing: {100/2**len(sol_str)}%")


######################################################################
# Energy Landscape
# ~~~~~~~~~~~~~~~~
# 
# For the case where there is just one layer on the QAOA, we can visualize
# the energy expectation value :math:`\langle H(z) \rangle` for the
# knapsack problem. The Figure below shows the landscape for the Knapsack
# problem with the optimal solution of the optimization step. The colormap
# is associated with the expectation energy, a red color meaning a higher
# energy (those regions we want to avoid).
# 
# Let’s start by iterating over different values of :math:`\beta` and
# :math:`\gamma`:
# 

n1 = n2 = 50
gammas = np.linspace(-np.pi/2, np.pi/2, n1)
betas = np.linspace(-np.pi/2, np.pi/2, n2)

landscape = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        landscape[i,j] = cost_function([gammas[i], betas[j]], ising_Hamiltonian)

fig, ax = plt.subplots(figsize=(5,5))
ax1 = ax.imshow(landscape, cmap="coolwarm", extent=[-np.pi/2, np.pi/2, np.pi/2, -np.pi/2])
ax.plot(sol.x[1], sol.x[0], marker="*", markersize=10, markeredgecolor="black", color="tab:red", label="optimal", linewidth=0)

ax.set_xticks([-np.pi/2, 0, np.pi/2])
ax.set_yticks([-np.pi/2, 0,np.pi/2])
ax.set_xticklabels([r"$-\pi/2$", 0, r"$\pi/2$"])
ax.set_yticklabels([r"$-\pi/2$", 0, r"$\pi/2$"])
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\gamma$")
ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.27))
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
# .. math::
# 
# 
#    x_1 + x_3 + x_5 = 1
# 
# 1. Repeat the steps in `Example <#KP>`__ and `QAOA <#qaoa>`__ for this
#    problem (note that you should modify the qubo formulation to include
#    the equality constraint.)
# 

def knapsack_new(values, weights, max_weight):
    n_items = len(values) # number of variables
    n_slacks = int(np.ceil(np.log2(max_weight))) # number of slack variables

    x = {i: Symbol(f"x{i}") for i in range(n_items)}  # variables that represent the items
    S = sum(2**k * Symbol(f"s{k}") for k in range(n_slacks)) # the slack variable in binary representation

    # objective function --------
    cost_fun = - sum([values[i]*x[i] for i in x]) # maximize the value of the items trasported Eq.12
    #(Note that minimizing the negative of cost function is the same that maximizing it)

    # ---------    constraint   Eq. 14  ----------
    constraint1 = max_weight - sum(weights[i] * x[i] for i in x) - S #inequality constraint

    constraint2 = x[1]  - 1

    cost = cost_fun + 2 * constraint1 ** 2  + 10 * constraint2 ** 2# Eq. 15 cost function with penalization term for the Knapsack problem
    return cost

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/alejandro_montanez.txt  