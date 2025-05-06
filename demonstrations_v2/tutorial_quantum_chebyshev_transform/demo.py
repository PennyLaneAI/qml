r"""Intro to the Quantum Chebyshev
=============================================================

This demo is inspired by the paper `"Quantum Chebyshev transform: mapping, embedding, learning and sampling distributions" <https://arxiv.org/abs/2306.17026>`__ wherein the authors describe a workflow for quantum Chebyshev-based model building. 
They demonstrate the use of the Chebyshev basis space in generative modeling for probability distributions. Crucial to their implementation of learning models in Chebyshev space is the quantum Chebyshev transform (QChT), which is used to swap between the computational basis and the Chebyshev basis. 

We will start by discussing Chebyshev polynomials and why you may want to work in Chebyshev space. Then we will show how the QChT can be implemented in PennyLane. 


.. figure:: ../_static/demonstration_assets/qft/socialthumbnail_large_QFT_2024-04-04.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

What are Chebyshev polynomials?
---------------------------------------

`Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials#As_a_basis_set>` of the first kind :math:`T_n(x)` are a set of orthogonal polynomials that are complete on the interval :math:`[-1,1]`. They can be defined as 

.. math::
  T_n(x) \equiv \cos(n \arccos(x))

where :math:`n` is the order of the polynomial. We can write out the first few orders explicitly.
.. math::
  T_0(x) &= 1 \\
  T_1(x) &= x \\
  T_2(x) &= 2x^2-1 \\
  T_3(x) &= 4x^3 - 3x \\
  T_4(x) &= 8x^4 - 8x^2 + 1 \\
  &\ \,\vdots \\
  T_{n+1}(x) &= 2xT_n(x) - T_{n-1}(x)

The recursion relation in the last line can be used to compute the next orders. Observe that odd and even order :math:`T_n` are odd and even functions, respectively. 
The roots of the :math:`T_n(x)` occur at the values 
.. math::
  x_n^\mathrm{Ch} = \cos\left(\frac{2k+1}{2n}\pi\right)\,, \quad k=0, ..., n-1\,. 

These are known as the `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`.

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/chebyshev_polynomials.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

The nodes are plotted above along with the corresponding polynomials. Note that the polynomials are normalized such that $T_n(1)=1$, and they satisfy a discrete orthogonality condition on the nodes of $T_N(x)$ in the following way for $k$, $\ell<N$
.. math::
  \sum^{N-1}_{j=0}T_k(x_j^\mathrm{Ch})T_\ell(x_j^\mathrm{Ch}) =  
    \begin{cases}
      0 & k \neq \ell\,,\\
      N & k = \ell = 0\,,\\
      N/2 & k = \ell \neq 0
    \end{cases}

The Chebyshev polynomials have a lot of \textit{nice} properties. Because they are complete, any function :math:`f(x)` on the interval :math:`x\in [-1,1]` can be expanded in :math:`T_n(x)` up to order :math`N` as :math:`f(x) = \sum_{j=0}^N a_j T_j(x)`. \
To do this process numerically for a discrete set of sampling points would take :math:`\mathcal{O}(N^2)` operations for a general set of complete functions \emoji{snail}. 
However, because of the way the Chebyshev polynomials are defined in terms of cosine, the `discrete Chebyshev transformation (DChT) <https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform>` can be related to the `discrete cosine transform (DCT) <https://en.wikipedia.org/wiki/Discrete_cosine_transform>` to leverage the efficiency of the `fast-Fourier-transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`-style algorithms, which take :math:`\mathcal{O}(N \log N)` operations \emoji{rocket}.

The DChT is sampled on the nodes of :math:`T_N(x)`, which are non-equidistant on the :math:`[-1,1]` interval. 
This non-uniform sampling has more resolution at the boundary, but less in the middle. This can be a benefit if you are, for example, solving a differential equation and expect more interesting features at the boundary, so the extra resolution there is useful. 
In general, working in the Chebyshev basis can have advantages over the Fourier basis for polynomial decomposition.

Quantum Chebyshev basis
---------------------------------------

"""

#
# About the authors
# -----------------
#
