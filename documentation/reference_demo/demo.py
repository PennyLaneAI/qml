r""" Reference Demo - a.k.a. the kitchen sink
=============================================

All the restructured text directives we support appear in this demo. If you find a new one, please add it here.

Complete list of reStructuredText directive types covered in this demo:  

5. math
3. figure
6. meta
9. related
7. note
11. tip
13. warning
8. raw
1. admonition

Unsupported directives:  

4. image(7) -> change to figure  
12. topic (1) -> change to admonition with class note(?)  
2. container (3) -> Doesn't work. Remove?  
10. role (5) -> Remove?  
"""

# %%
# Comprehensive Mathematical Notation Reference
# ---------------------------------------------
#
# This document contains a comprehensive collection of all mathematical symbols and notations
# used across the PennyLane quantum machine learning demonstrations.

# %%
# QUANTUM MECHANICS NOTATION
# **************************

# %%
# Quantum states and bra-ket notation  
#
# .. math::
#    
#     |\psi\rangle, \quad |\phi\rangle, \quad |0\rangle, \quad |1\rangle, \quad |+\rangle, \quad |-\rangle
#     \langle\psi|, \quad \langle\phi|, \quad \langle 0|, \quad \langle 1|, \quad \langle +|, \quad \langle -|
#     \langle\psi|\phi\rangle, \quad \langle\psi|\hat{O}|\phi\rangle, \quad |\psi\rangle\langle\phi|

# %%    
# Quantum operators and matrices  
#
# .. math::
#    
#     \hat{H}, \quad \hat{O}, \quad \hat{A}, \quad \hat{B}, \quad \hat{U}, \quad \hat{V}
    
# %%
# Pauli operators
#
# .. math::
#    
#     \sigma_x, \quad \sigma_y, \quad \sigma_z, \quad X, \quad Y, \quad Z, \quad I  
    
# %%
# Multi-qubit operators
#
# .. math::
#    
#     X_0, \quad Y_1, \quad Z_2, \quad X_i Y_j Z_k  
    
# %%
# Tensor products
#
# .. math::
#    
#     |\psi\rangle \otimes |\phi\rangle, \quad A \otimes B, \quad \bigotimes_{i=1}^n \sigma_i  
    
# %%
# Quantum gates and unitaries
#
# .. math::
#    
#     U(\theta), \quad R_x(\theta), \quad R_y(\phi), \quad R_z(\gamma)
#     CNOT, \quad CZ, \quad \text{Toffoli}, \quad H  
    
# %%
# Density matrices and mixed states
#
# .. math::
#    
#     \rho, \quad \rho_0, \quad \rho(\theta), \quad \text{Tr}(\rho), \quad \text{Tr}(\rho A)
    
# %%
# Time evolution
#
# .. math::
#    
#     U(t) = e^{-i\hat{H}t/\hbar}, \quad |\psi(t)\rangle = U(t)|\psi(0)\rangle   
    
# %%
# GREEK LETTERS
# **************

# %%
# Lowercase Greek
#
# .. math::
#    
#     \alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta,
#     \iota, \kappa, \lambda, \mu, \nu, \xi, \pi, \varpi, \rho, \varrho, \sigma, \varsigma,
#     \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega
    
# %%
# Uppercase Greek
#
# .. math::
#    
#     \Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Upsilon, \Phi, \Psi, \Omega    
    
# %%
# MATHEMATICAL OPERATORS
# **********************
    
# %%
# Summation and products
#
# .. math::
#    
#     \sum_{i=1}^n, \quad \sum_{i,j}, \quad \prod_{k=0}^{N-1}, \quad \bigcup_{i}, \quad \bigcap_{j}   
    
# %%
# Integrals
#
# .. math::
#    
#     \int_0^T, \quad \int_{-\infty}^{\infty}, \quad \oint, \quad \iint, \quad \iiint    
    
# %%
# Derivatives and gradients
#
# .. math::
#    
#     \frac{\partial}{\partial \theta}, \quad \frac{d}{dt}, \quad \nabla, \quad \nabla^2, \quad \partial_i, \quad \partial_t   
    
# %%
# Limits and infinity
#
# .. math::
#    
#     \lim_{n \to \infty}, \quad \infty, \quad -\infty, \quad \pm\infty    
    
# %%
# LINEAR ALGEBRA
# **************
    
# %%
# Vectors and matrices
#
# .. math::
#    
#     \vec{v}, \quad \boldsymbol{v}, \quad \mathbf{A}, \quad A^T, \quad A^\dagger, \quad A^{-1}, \quad A^*   
    
# %%
# Matrix operations
#
# .. math::
#    
#     \text{det}(A), \quad \text{tr}(A), \quad \text{rank}(A), \quad \|A\|, \quad \|v\|_2    

# %%
# Eigenvalues and eigenvectors
#
# .. math::
#    
#     A|\psi\rangle = \lambda|\psi\rangle, \quad \text{spec}(A), \quad \lambda_{\min}, \quad \lambda_{\max}    
    
# %%
# Inner and outer products
#
# .. math::
#    
#     \langle u, v \rangle, \quad u \cdot v, \quad u \times v, \quad |u\rangle\langle v|    

# %%
# COMPLEX NUMBERS AND FUNCTIONS
# ******************************
    
# %%
# Complex notation
#
# .. math::
#    
#     z = a + bi, \quad \text{Re}(z), \quad \text{Im}(z), \quad |z|, \quad z^*, \quad \bar{z}  
#     e^{i\theta} = \cos\theta + i\sin\theta, \quad e^{i\pi} = -1    
    
# %%
# Common functions
#
# .. math::
#    
#     \sin(x), \quad \cos(x), \quad \tan(x), \quad \sinh(x), \quad \cosh(x), \quad \tanh(x)  
#     \exp(x), \quad \log(x), \quad \ln(x), \quad \log_2(x)    
    
# %%
# PROBABILITY AND STATISTICS
# ***************************

# %%
# Probability notation
#
# .. math::
#    
#     P(A), \quad P(A|B), \quad P(A \cap B), \quad P(A \cup B)   
    
# %%
# Expectations and variance
#
# .. math::
#    
#     \mathbb{E}[X], \quad \text{Var}(X), \quad \text{Cov}(X,Y), \quad \text{Corr}(X,Y)   
    
# %%
# Distributions
#
# .. math::
#    
#     X \sim \mathcal{N}(\mu, \sigma^2), \quad X \sim \text{Bernoulli}(p)   
    
# %%
# SET THEORY AND LOGIC
# ********************

# %%
# Set operations
#
# .. math::
#    
#     A \in B, \quad A \subset B, \quad A \subseteq B, \quad A \cup B, \quad A \cap B, \quad A \setminus B
#     \emptyset, \quad \mathbb{N}, \quad \mathbb{Z}, \quad \mathbb{Q}, \quad \mathbb{R}, \quad \mathbb{C}
    
# %%
# Logic symbols
#
# .. math::
#    
#     \land, \quad \lor, \quad \neg, \quad \implies, \quad \iff, \quad \forall, \quad \exists   

# %%
# OPTIMIZATION AND CALCULUS
# **************************
    
# %%
# Optimization
#
# .. math::
#    
#     \min_{x}, \quad \max_{x}, \quad \arg\min_{x}, \quad \arg\max_{x}    
#     \theta^* = \arg\min_\theta \mathcal{L}(\theta)    
    
# %%
# Gradients and Hessians
#
# .. math::
#    
#     \nabla f(\mathbf{x}), \quad \mathbf{H}f(\mathbf{x}), \quad \frac{\partial^2 f}{\partial x_i \partial x_j}    
    
# %%
# Learning rates and updates
#
# .. math::
#    
#     \theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)   

# %%
# QUANTUM INFORMATION THEORY
#**************************
    
# %%
# Entropy and information
#
# .. math::
#    
#     S(\rho) = -\text{Tr}(\rho \log \rho), \quad H(X), \quad I(X:Y)    
    
# %%
# Fidelity and distance measures
#
# .. math::
#    
#     F(\rho, \sigma), \quad d(\rho, \sigma), \quad \|\rho - \sigma\|_1    
    
# %%
# Quantum channels
#
# .. math::
#    
#     \mathcal{E}(\rho), \quad \Phi(\rho), \quad \mathcal{N}(\rho)
    
# %%
# VARIATIONAL QUANTUM ALGORITHMS
# ******************************
    
# %%
# Cost functions
#
# .. math::
#    
#     \mathcal{L}(\theta) = \langle 0|U^\dagger(\theta) H U(\theta)|0\rangle    

# %%
# QAOA notation
#
# .. math::
#    
#     U_B(\beta) = e^{-i\beta B}, \quad U_C(\gamma) = e^{-i\gamma C}    
#     |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = U_B(\beta_p) U_C(\gamma_p) \cdots U_B(\beta_1) U_C(\gamma_1) |+\rangle
  
# %%
# VQE notation
#
# .. math::
#    
#     E_0 = \min_\theta \langle \psi(\theta)|H|\psi(\theta)\rangle   
    
# %%
# MACHINE LEARNING NOTATION
# **************************
    
# %%
# Data and parameters
#
# .. math::
#    
#     \mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad \theta \in \Theta, \quad f_\theta(x)   

# %%
# Loss functions
#
# .. math::
#    
#     \ell(y, \hat{y}), \quad \mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(y_i, f_\theta(x_i))
    
# %%
# Gradients and optimization
#
# .. math::
#    
#     \frac{\partial \mathcal{L}}{\partial \theta}, \quad \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}   

# %%
# QUANTUM CIRCUIT NOTATION
# **************************

# %%
# Quantum circuits
#
# .. math::
#    
#     U(\vec{\theta}) = \prod_{j=1}^L U_j(\theta_j), \quad V(\vec{x}) = \prod_{k=1}^K V_k(x_k)
    
# %%
# Measurements
#
# .. math::
#    
#     \langle M \rangle = \langle \psi|M|\psi\rangle, \quad p_m = |\langle m|\psi\rangle|^2
    
# %%
# SPECIAL SYMBOLS AND NOTATION
# ****************************
    
# %%
# Approximation and limits
#
# .. math::
#    
#     \approx, \quad \simeq, \quad \sim, \quad \propto, \quad \ll, \quad \gg, \quad O(n), \quad \Theta(n)    
   
# %%
# Inequalities
#
# .. math::
#    
#     \leq, \quad \geq, \quad <, \quad >, \quad \neq, \quad \equiv

# %%
# Geometric and topological
#
# .. math::
#    
#     \angle, \quad \perp, \quad \parallel, \quad \cong, \quad \triangle   

# %%
# Miscellaneous symbols
#
# .. math::
#    
#     \dagger, \quad \ast, \quad \circ, \quad \bullet, \quad \square, \quad \diamond, \quad \star   

# %%
# MATRIX DECOMPOSITIONS
# *********************
    
# %%
# SVD and eigendecomposition
#
# .. math::
#    
#     A = U\Sigma V^\dagger, \quad A = Q\Lambda Q^\dagger  
    
# %%
# QR and Cholesky
#
# .. math::
#    
#     A = QR, \quad A = LL^T
    
# %%
# QUANTUM ERROR CORRECTION
# ************************
    
# %%
# Error syndromes and stabilizers
#
# .. math::
#    
#     S_i, \quad \text{stab}(\mathcal{C}), \quad [n,k,d]
    
# %%
# Logical operators
#
# .. math::
#    
#     \bar{X}, \quad \bar{Z}, \quad |\bar{0}\rangle, \quad |\bar{1}\rangle
    

# %%
# HAMILTONIAN SIMULATION
# ************************
    
# %%
# Trotterization
#
# .. math::
#    
#     e^{-i(A+B)t} \approx \left(e^{-iAt/n}e^{-iBt/n}\right)^n
    
# %%
# Pauli decomposition
#
# .. math::
#    
#     H = \sum_{P \in \mathcal{P}_n} h_P P, \quad \mathcal{P}_n = \{I,X,Y,Z\}^{\otimes n}

# %%
# FOURIER ANALYSIS
# ****************

# %%
# Fourier transform
#
# .. math::
#    
#     \hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt
    
# %%
# Discrete Fourier transform
#
# .. math::
#    
#     \tilde{f}_k = \sum_{j=0}^{N-1} f_j e^{-2\pi i jk/N}
    
    
# %%
# ADVANCED NOTATIONS
# ******************
    
# %%
# Lie algebras and groups
#
# .. math::
#    
#     [A,B] = AB - BA, \quad \text{ad}_A(B) = [A,B], \quad e^A B e^{-A}
    
# %%
# Variational derivatives
#
# .. math::
#    
#     \frac{\delta}{\delta f}, \quad \delta f, \quad \mathcal{F}[f] 
    
# %%
# Asymptotic notation
#
# .. math::
#    
#     f(n) = O(g(n)), \quad f(n) = \Omega(g(n)), \quad f(n) = \Theta(g(n))
    
# %%
# FIGURES
# -------

# %%
# Figure
#
# .. figure:: ../_static/demonstration_assets/period_finding/periodic_function.png
#     :align: center
#     :width: 90%
#     :scale: 100%
#     :alt: Example of a figure directive. 
#     :target: javascript:void(0);
#     :figwidth: 80%
#    
#     Figure 2. Example of a discrete periodic function f(x) over the integers x = 0,...,11. 
#     The function only takes the same value when moving exactly 4 integers on the x-axis. Note:
#     not sure what to do for this one. Should we make a fake image for this demo?

# %%
# METADATA AND RELATED CONTENT
# ----------------------------
#
# These shouldn't appear in the rendered demos.

# %%
# .. meta::
#     :property="og:description": Example of a meta directive.
#     :property="og:image": ../_static/demonstration_assets/period_finding/periodic_function.png
#
# .. related::
#     tutorial_qft Quantum Fourier Transform
#     tutorial_qft_arithmetics Quantum Fourier Transform Arithmetic

# %%
# RAW (USUALLY FOR HTML)
# ----------------------
#
# This should be handled natively by pandoc and should appear in the rendered demos.

# %%
# .. raw:: html
#
#     <center>
#         <div class="alert alert-warning">
#             Example of a raw directive.
#         </div>
#     </center>

# %%
# NOTES, TIPS, WARNINGS, AND ADMONITIONS
# --------------------------------------
#
# All of these are unsupported by pandoc. A custom filter is used to convert these to epigraphs (BlockQutoes in markdown).

# %%
# .. note::
#     Example of a note directive.

# %%
# .. tip::
#     Example of a tip directive.

# %%
# .. warning::
#     Example of a warning directive.

# %%
# .. admonition:: This is an admonition.
#     :class: note
#
#     Example of an admonition directive.

# %%
# Links
# -----
#
# Example of an `external link <https://pennylane.ai/qml>`_.
#
# Example of an :doc:`InterSphinx demo link <demos/tutorial_liealgebra>`.
#
# Example of an :doc:`Intersphinx doc link <introduction/operations>`.
#
# Example of a PennyLane function link: :func:`~.pennylane.ops.one_qubit_decomposition`.
#
# Example of a PennyLane class link: :class:`~.pennylane.ops.RX`.
