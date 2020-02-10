# coding=utf-8
r'''
VQE with parallel QPUs on Rigetti-Forest
========================================

This tutorial showcases how using asynchronously evaluated parallel QPUs can speed up calculation
of the potential energy surface of molecular hydrogen.

Using a VQE setup, we task two devices from the
[PennyLane-Forest](https://pennylane-forest.readthedocs.io/en/latest/) plugin with evaluating
separate terms in the qubit Hamiltonian of $H_2$. As these devices are allowed to operate
asynchronously, i.e., at the same time and without having to wait for each other,
the calculation can be performed in half the time.

We begin by importing the prerequisite libraries:
'''

import pennylane as qml
from pennylane import qchem

import matplotlib.pyplot as plt
import numpy as np
import time

##############################################################################
# The ```pennylane-qchem``` module is installed separately using ```pip install
# pennylane-qchem```. This tutorial also requires the ```pennylane-forest``` package, which can
# be installed using ```pip install pennylane-forest```.
#
# Finding the qubit Hamiltonians of $H_{2}$
# -----------------------------------------
#
# The objective of this tutorial is to evaluate the potential energy surface of molecular
# hydrogen. This is achieved by finding the ground state energy of $H_{2}$ for a range of atomic
# separations.
#
# Each atomic separation results in a different qubit Hamiltonian. To find the corresponding
# Hamiltonian, we use the ```generate_hamiltonian``` function of the ``qchem`` package. Further
# details on the mapping from electronic Hamiltonian of a molecule to a qubit Hamiltonian can be
# found in the :ref:`Quantum Chemistry with PennyLane <qchem-beginner>` and
# :ref:`Quantum Chemistry with PennyLane <qchem-implementations>` tutorials.
#
# We begin by creating a dictionary containing molecular separations and corresponding data files
# saved in ```.xyz``` format.

data = {  # keys: atomic separations (ångström), values: corresponding files
    0.3: 'xyz/h2_0.30.xyz',
    0.5: 'xyz/h2_0.50.xyz',
    0.7: 'xyz/h2_0.70.xyz',
    0.9: 'xyz/h2_0.90.xyz',
    1.1: 'xyz/h2_1.10.xyz',
    1.3: 'xyz/h2_1.30.xyz',
    1.5: 'xyz/h2_1.50.xyz',
    1.7: 'xyz/h2_1.70.xyz',
    1.9: 'xyz/h2_1.90.xyz',
    2.1: 'xyz/h2_2.10.xyz',
}

##############################################################################
# The next step is to create the qubit Hamiltonians for each value of molecular separation.

hamiltonians = [
    qchem.generate_hamiltonian(
        mol_name=str(separation),
        mol_geo_file=file,
        mol_charge=0,
        multiplicity=1,
        basis_set='sto-3g',
        n_active_electrons=2,
        n_active_orbitals=2,
    )[0]
    for separation, file in data.items()]

##############################################################################
# Each Hamiltonian can be written as a linear combination of fifteen tensor products of Pauli
# matrices.

[len(h.ops) for h in hamiltonians]

##############################################################################
# Defining the energy function
# ----------------------------
#
# The fifteen Pauli terms comprising each Hamiltonian can conventionally be evaluated in a
# sequential manner: we evaluate one expectation value at a time before moving on to the next.
# However, this task is highly suited to parallelization. With access to multiple QPUs,
# we can split up evaluating the terms between the QPUs and gain an increase in processing speed.
#
# Note that some of the Pauli terms commute, and so they can be evaluated in practice with fewer
# than fifteen quantum circuit runs. Nevertheless, these quantum circuit runs can still be
# parallelized to multiple QPUs.
#
# Let's suppose we have access to two QPUs: ```Aspen-4-4Q-E``` and ```Aspen-7-4Q-C``` from
# Rigetti. We can evaluate the expectation value of each Hamiltonian with eight terms run on
# ```Aspen-4-4Q-D``` and seven terms run on ```Aspen-7-4Q-C```.
#
# To do this, start by instantiating a device for each term:

# Use forest.qpu if hardware access is available
devs_4 = [qml.device('forest.qvm', device="Aspen-4-4Q-D") for _ in range(8)]
devs_7 = [qml.device('forest.qvm', device="Aspen-7-4Q-C") for _ in range(7)]
devs = devs_4 + devs_7

##############################################################################
# We must also define a circuit to prepare the ground state. The simple circuit below is able to
# accurately capture the ground state encoded into qubits. It has a single free parameter setting
# a Y-rotation on the third qubit.

def circuit(param, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
    qml.RY(param, wires=2)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

##############################################################################
# Each molecular separation requires a different Y-rotation parameter. The values of these
# parameters can be found by minimizing the ground state energy as outlined in
# :ref:`Quantum Chemistry with PennyLane <qchem-implementations>`. In this tutorial we load
# pre-optimized rotations and focus on comparing the speed of evaluating the potential energy
# surface with sequential and parallel evaluation.

params = np.load('RY_params.npy')

##############################################################################
# Finally, the energies as a function of rotation angle can be given using ```VQECost```.

energies = [qml.VQECost(circuit, h, devs) for h in hamiltonians]

##############################################################################
# Calculating the potential energy surface
# ----------------------------------------
#
# ```VQECost``` returns a ```QNodeCollection``` which can be evaluated based using the input
# parameters to the ansatz circuit. As of ```v0.8.0```, the ```QNodeCollection``` can be
# evaluated asynchronously by specifying the ```parallel = True``` keyword argument. With the
# default ```parallel = False```, the QNodes are evaluated one at a time.
#
# We can use this feature to compare the sequential and parallel times to calculate the potential
# energy surface. The following function calculates the surface:

def calculate_surface(parallel=True):
    t0 = time.time()
    surface = []

    for i, e in enumerate(energies):
        print('Running for molecular separation {} Å'.format(list(data.keys())[i]))
        s = e(params[i], parallel=parallel)
        surface.append(s)

    t1 = time.time()

    print('Evaluation time: {0:.2f} s'.format(t1 - t0))
    return(surface)

print('Evaluating the potential energy surface sequentially')
surface = calculate_surface(False)

print('\nEvaluating the potential energy surface in parallel')
surface = calculate_surface(True)

##############################################################################
# We can finally plot the potential energy surface.

plt.plot(surface, linewidth=2.2, marker='o')
plt.title('Potential energy surface for molecular hydrogen', fontsize=12)
plt.xlabel('Atomic separation (Å)', fontsize=16)
plt.ylabel('Ground state energy (Ha)', fontsize=16)
plt.grid(True)
