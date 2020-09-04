.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_app_tutorial_vqe.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_app_tutorial_vqe.py:


A brief overview of VQE
=======================

The Variational Quantum Eigensolver (VQE) :ref:`[1, 2]<vqe_references>` is a flagship algorithm for
quantum chemistry using near-term quantum computers. VQE is an application of the `Ritz variational
principle <https://en.wikipedia.org/wiki/Ritz_method>`_  where a quantum computer is used to 
prepare a wave function ansatz of the molecule and estimate the expectation value of its electronic
Hamiltonian while a classical optimizer is used to adjust the quantum circuit parameters in order 
to find the molecule's ground state energy.

For example, if we use a minimal basis, the ground state wave function of the hydrogen molecule 
:math:`\vert \Psi \rangle = \alpha \vert 1100 \rangle + \beta \vert 0011 \rangle` consists of only
the Hartree-Fock component and a doubly-excited configuration where the two electrons occupy the 
highest-energy molecular orbitals. If we use a quantum computer to prepare the four-qubit 
entangled state :math:`\vert \Psi \rangle`, the ultimate goal of the VQE algorithm 
is to find the values of :math:`\alpha` and :math:`\beta` that minimize the expectation value of 
the electronic Hamiltonian.
 
The PennyLane library allows users to implement the full VQE algorithm using only a few
lines of code. In this tutorial, we guide you through a calculation of the ground-state energy of
the hydrogen molecule. Let's get started! ⚛️

Building the electronic Hamiltonian
-----------------------------------

The first step is to import the required libraries and packages:

.. code-block:: default


    import pennylane as qml
    from pennylane import numpy as np







The second step is to specify the molecule whose properties we aim to calculate.
This is done by providing three pieces of information: the geometry and charge of the molecule,
and the spin multiplicity of the electronic configuration.

The geometry of a molecule is given by the three-dimensional coordinates and symbols of all
its atomic species. There are several databases such as `the NIST Chemistry
WebBook <https://webbook.nist.gov/chemistry/name-ser/>`_, `ChemSpider <http://www.chemspider.com/>`_
and `SMART-SNS <http://smart.sns.it/molecules/>`_ that provide
geometrical data for a large number of molecules. Here, we make use of a locally saved file in
``.xyz`` format that contains the geometry of the hydrogen molecule, and specify its name for
later use:


.. code-block:: default


    geometry = 'h2.xyz'







Alternatively, you can download the file here: :download:`h2.xyz </implementations/h2.xyz>`.

The charge determines the number of electrons that have been added or removed compared to the
neutral molecule. In this example, as is the case in many quantum chemistry simulations,
we will consider a neutral molecule:


.. code-block:: default


    charge = 0







It is also important to define how the electrons occupy the molecular orbitals to be optimized
within the `Hartree-Fock approximation <https://en.wikipedia.org/wiki/Hartree-Fock_method>`__. 
This is captured by the `multiplicity <https://en.wikipedia.org/wiki/Multiplicity_(chemistry)>`_ 
parameter, which is related to the number of unpaired electrons in the Hartree-Fock state. For 
the neutral hydrogen molecule, the multiplicity is one:


.. code-block:: default


    multiplicity = 1







Finally, we need to specify the `basis set <https://en.wikipedia.org/wiki/Basis_set_(
chemistry)>`_ used to approximate atomic orbitals. This is typically achieved by using a linear
combination of Gaussian functions. In this example, we will use the minimal basis STO-3g where a
set of 3 Gaussian functions are contracted to represent an atomic Slater-type orbital (STO):


.. code-block:: default


    basis_set = 'sto-3g'







At this stage, to compute the molecule's Hamiltonian in the Pauli basis, several
calculations need to be performed. With PennyLane, these can all be done in a
single line by calling the function :func:`~.pennylane_qchem.qchem.generate_hamiltonian`. The
first input to the function is a string denoting the name of the molecule, which will determine
the name given to the saved files that are produced during the calculations:


.. code-block:: default


    name = 'h2'







The geometry, charge, multiplicity, and basis set must also be specified as input. Finally,
the number of active electrons and active orbitals have to be indicated, as well as the
fermionic-to-qubit mapping, which can be either Jordan-Wigner (``jordan_wigner``) or Bravyi-Kitaev
(``bravyi_kitaev``). The outputs of the function are the qubit Hamiltonian of the molecule and the
number of qubits needed to represent it:


.. code-block:: default


    h, nr_qubits = qml.qchem.generate_hamiltonian(
        name,
        geometry,
        charge,
        multiplicity,
        basis_set,
        n_active_electrons=2,
        n_active_orbitals=2,
        mapping='jordan_wigner'
    )

    print('Number of qubits = ', nr_qubits)
    print('Hamiltonian is ', h)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Number of qubits =  4
    Hamiltonian is  (-0.04207897647782188) [I0]
    + (0.17771287465139934) [Z0]
    + (0.1777128746513993) [Z1]
    + (-0.24274280513140484) [Z2]
    + (-0.24274280513140484) [Z3]
    + (0.17059738328801055) [Z0 Z1]
    + (0.04475014401535161) [Y0 X1 X2 Y3]
    + (-0.04475014401535161) [Y0 Y1 X2 X3]
    + (-0.04475014401535161) [X0 X1 Y2 Y3]
    + (0.04475014401535161) [X0 Y1 Y2 X3]
    + (0.12293305056183801) [Z0 Z2]
    + (0.1676831945771896) [Z0 Z3]
    + (0.1676831945771896) [Z1 Z2]
    + (0.12293305056183801) [Z1 Z3]
    + (0.176276408043196) [Z2 Z3]


That's it! From here on, we can use PennyLane as usual, employing its entire stack of
algorithms and optimizers.

Implementing the VQE algorithm
------------------------------

PennyLane contains the :class:`~.pennylane.VQECost` class, specifically
built to implement the VQE algorithm. We begin by defining the device, in this case a simple
qubit simulator:


.. code-block:: default


    dev = qml.device('default.qubit', wires=nr_qubits)







In VQE, the goal is to train a quantum circuit to prepare the ground state of the input
Hamiltonian. This requires a clever choice of circuit, which should be complex enough to
prepare the ground state, but also sufficiently easy to optimize. In this example, we employ a
variational circuit that is capable of preparing the normalized states of the form 
:math:`\alpha|1100\rangle + \beta|0011\rangle` which encode the ground state wave function of 
the hydrogen molecule described with a minimal basis set. The circuit consists of single-qubit 
rotations on all wires, followed by three entangling CNOT gates, as shown in the figure below:

|

.. figure:: /implementations/variational_quantum_eigensolver/sketch_circuit.png
    :width: 50%
    :align: center

|


In the circuit, we apply single-qubit rotations, followed by CNOT gates:


.. code-block:: default



    def circuit(params, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
        for i in wires:
            qml.Rot(*params[i], wires=i)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])







.. note::

    The qubit register has been initialized to :math:`|1100\rangle` which encodes the
    Hartree-Fock state of the hydrogen molecule described with a `minimal basis
    <https://en.wikipedia.org/wiki/Basis_set_(chemistry)#Minimal_basis_sets>`__.

The cost function for optimizing the circuit can be created using the :class:`~.pennylane.VQECost`
class, which is tailored for VQE optimization. It requires specifying the
circuit, target Hamiltonian, and the device, and returns a cost function that can
be evaluated with the circuit parameters:


.. code-block:: default



    cost_fn = qml.VQECost(circuit, h, dev)








Wrapping up, we fix an optimizer and randomly initialize circuit parameters. For reliable
results, we fix the seed of the random number generator, since in practice it may be necessary
to re-initialize the circuit several times before convergence occurs.


.. code-block:: default


    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    np.random.seed(0)
    params = np.random.normal(0, np.pi, (nr_qubits, 3))

    print(params)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 5.54193389  1.25713095  3.07479606]
     [ 7.03997361  5.86710646 -3.07020901]
     [ 2.98479079 -0.47550269 -0.32427159]
     [ 1.28993324  0.45252622  4.56873497]]


We carry out the optimization over a maximum of 200 steps, aiming to reach a convergence
tolerance (difference in cost function for subsequent optimization steps) of :math:`\sim 10^{
-6}`.


.. code-block:: default


    max_iterations = 200
    conv_tol = 1e-06

    prev_energy = cost_fn(params)
    for n in range(max_iterations):
        params = opt.step(cost_fn, params)
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)

        if n % 20 == 0:
            print('Iteration = {:},  Ground-state energy = {:.8f} Ha,  Convergence parameter = {'
                  ':.8f} Ha'.format(n, energy, conv))

        if conv <= conv_tol:
            break

        prev_energy = energy

    print()
    print('Final convergence parameter = {:.8f} Ha'.format(conv))
    print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
    print('Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)'.
            format(np.abs(energy - (-1.136189454088)), np.abs(energy - (-1.136189454088))*627.503))
    print()
    print('Final circuit parameters = \n', params)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Iteration = 0,  Ground-state energy = -0.88179557 Ha,  Convergence parameter = 0.07432580 Ha
    Iteration = 20,  Ground-state energy = -1.13380513 Ha,  Convergence parameter = 0.00043673 Ha
    Iteration = 40,  Ground-state energy = -1.13558756 Ha,  Convergence parameter = 0.00001950 Ha
    Iteration = 60,  Ground-state energy = -1.13585794 Ha,  Convergence parameter = 0.00000993 Ha
    Iteration = 80,  Ground-state energy = -1.13600617 Ha,  Convergence parameter = 0.00000553 Ha
    Iteration = 100,  Ground-state energy = -1.13608848 Ha,  Convergence parameter = 0.00000306 Ha
    Iteration = 120,  Ground-state energy = -1.13613394 Ha,  Convergence parameter = 0.00000169 Ha

    Final convergence parameter = 0.00000099 Ha
    Final value of the ground-state energy = -1.13615709 Ha
    Accuracy with respect to the FCI energy: 0.00003237 Ha (0.02031093 kcal/mol)

    Final circuit parameters = 
     [[ 5.54193389e+00  1.30219523e-08  3.07479606e+00]
     [ 7.03997361e+00  6.28318530e+00 -3.07020901e+00]
     [ 2.98479079e+00 -2.09540998e-01 -4.16893297e-02]
     [ 1.28993324e+00  1.30898301e-12  4.56873497e+00]]


Success! 🎉🎉🎉 The ground-state energy of the hydrogen molecule has been estimated with chemical
accuracy (< 1 kcal/mol) with respect to the exact value of -1.136189454088 Hartree (Ha) obtained
from a full configuration-interaction (FCI) calculation. This is because, for the optimized 
values of the single-qubit rotation angles, the state prepared by the VQE ansatz is precisely
the FCI ground-state of the :math:`H_2` molecule :math:`|H_2\rangle_{gs} = 0.99 |1100\rangle - 0.10
|0011\rangle`.

What other molecules would you like to study using PennyLane?

.. _vqe_references:

References
----------

1. Alberto Peruzzo, Jarrod McClean *et al.*, "A variational eigenvalue solver on a photonic
   quantum processor". `Nature Communications 5, 4213 (2014).
   <https://www.nature.com/articles/ncomms5213?origin=ppub>`__

2. Yudong Cao, Jonathan Romero, *et al.*, "Quantum Chemistry in the Age of Quantum Computing".
   `Chem. Rev. 2019, 119, 19, 10856-10915. 
   <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  27.576 seconds)


.. _sphx_glr_download_app_tutorial_vqe.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_vqe.py <tutorial_vqe.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_vqe.ipynb <tutorial_vqe.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
