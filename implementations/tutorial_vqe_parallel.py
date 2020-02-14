# coding=utf-8
r"""
VQE with parallel QPUs on Rigetti Forest
========================================

This tutorial showcases how using asynchronously-evaluated parallel QPUs can speed up the
calculation of the potential energy surface of molecular hydrogen (:math:`H_2`).

Using a VQE setup, we task two devices from the
`PennyLane-Forest <https://pennylane-forest.readthedocs.io/en/latest/>`__ plugin with evaluating
separate terms in the qubit Hamiltonian of :math:`H_2`. As these devices are allowed to operate
asynchronously, i.e., at the same time and without having to wait for each other,
the calculation can be performed in roughly half the time.

We begin by importing the prerequisite libraries:
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import qchem

##############################################################################
# This tutorial requires the ``pennylane-qchem``, ``pennylane-forest`` and ``dask``
# packages, which are installed separately using:
#
# .. code-block:: bash
#
#    pip install pennylane-qchem
#    pip install pennylane-forest
#    pip install "dask[delayed]"
#
# Finding the qubit Hamiltonians of :math:`H_{2}`
# -----------------------------------------------
#
# The objective of this tutorial is to evaluate the potential energy surface of molecular
# hydrogen. This is achieved by finding the ground state energy of :math:`H_{2}` as we increase
# the bond length between the hydrogen atoms.
#
# Each inter-atomic distance results in a different qubit Hamiltonian. To find the corresponding
# Hamiltonian, we use the :func:`~.pennylane_qchem.qchem.generate_hamiltonian` function of the
# :mod:`~.pennylane_qchem.qchem` package. Further details on the mapping from the electronic
# Hamiltonian of a molecule to a qubit Hamiltonian can be found in the
# :doc:`../tutorial/tutorial_quantum_chemistry` and :doc:`../app/tutorial_vqe`
# tutorials.
#
# We begin by creating a dictionary containing a selection of bond lengths and corresponding data
# files saved in `XYZ <https://en.wikipedia.org/wiki/XYZ_file_format>`__ format. These files
# follow a standard format for specifying the geometry of a molecule and can be downloaded as a
# Zip from :download:`here <../implementations/vqe_parallel/vqe_parallel.zip>`.

data = {  # keys: atomic separations (in Angstroms), values: corresponding files
    0.3: "vqe_parallel/h2_0.30.xyz",
    0.5: "vqe_parallel/h2_0.50.xyz",
    0.7: "vqe_parallel/h2_0.70.xyz",
    0.9: "vqe_parallel/h2_0.90.xyz",
    1.1: "vqe_parallel/h2_1.10.xyz",
    1.3: "vqe_parallel/h2_1.30.xyz",
    1.5: "vqe_parallel/h2_1.50.xyz",
    1.7: "vqe_parallel/h2_1.70.xyz",
    1.9: "vqe_parallel/h2_1.90.xyz",
    2.1: "vqe_parallel/h2_2.10.xyz",
}

##############################################################################
# The next step is to create the qubit Hamiltonians for each value of the inter-atomic distance.

hamiltonians = [
    qchem.generate_hamiltonian(
        mol_name=str(separation),
        mol_geo_file=file,
        mol_charge=0,
        multiplicity=1,
        basis_set="sto-3g",
    )[0]  # We take the zero element (the Hamiltonian). The other element is the qubit number
    for separation, file in data.items()
]

##############################################################################
# Each Hamiltonian can be written as a linear combination of fifteen tensor products of Pauli
# matrices. Let's take a look more closely at one of the Hamiltonians:

h = hamiltonians[0]

print('Number of terms: {}\n'.format(len(h.ops)))
for op in h.ops:
    print('Measurement {} on wires {}'.format(op.name, op.wires))

##############################################################################
# Defining the energy function
# ----------------------------
#
# The fifteen Pauli terms comprising each Hamiltonian can conventionally be evaluated in a
# sequential manner: we evaluate one expectation value at a time before moving on to the next.
# However, this task is highly suited to parallelization. With access to multiple QPUs,
# we can split up evaluating the terms between the QPUs and gain an increase in processing speed.
#
#
# .. note::
#    Some of the Pauli terms commute, and so they can be evaluated in practice with fewer than
#    fifteen quantum circuit runs. Nevertheless, these quantum circuit runs can still be
#    parallelized to multiple QPUs.
#
# Let's suppose we have access to two QPUs: ``Aspen-4-4Q-E`` and ``Aspen-7-4Q-D`` from
# Rigetti. We can evaluate the expectation value of each Hamiltonian with eight terms run on
# ``Aspen-4-4Q-E`` and seven terms run on ``Aspen-7-4Q-D``, as summarized by the diagram below:
#
# .. figure:: /implementations/vqe_parallel/diagram.png
#    :width: 65%
#    :align: center
#
# To do this, start by instantiating a device for each term:

devs_4 = [qml.device("forest.qvm", device="Aspen-4-4Q-E") for _ in range(8)]
devs_7 = [qml.device("forest.qvm", device="Aspen-7-4Q-D") for _ in range(7)]
devs = devs_4 + devs_7

##############################################################################
# .. note::
#     You can swap out ``forest.qvm`` for ``forest.qpu`` if hardware access is available.
#
# We must also define a circuit to prepare the ground state, which is a superposition of the
# Hartree-Fock (:math:`|1100\rangle`) and doubly-excited (:math:`|0011\rangle`) configurations.
# The simple circuit below is able to prepare states of the form :math:`\alpha |1100\rangle +
# \beta |0011\rangle` and hence encode the ground state wave function of the hydrogen molecule. The
# circuit has a single free parameter, which controls a Y-rotation on the third qubit.


def circuit(param, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
    qml.RY(param, wires=2)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])


##############################################################################
# The ground state for each inter-atomic distance is characterized by a different Y-rotation angle.
# The values of these Y-rotations can be found by minimizing the ground state energy as outlined in
# :doc:`../app/tutorial_vqe`. In this tutorial, we load pre-optimized rotations and focus on
# comparing the speed of evaluating the potential energy surface with sequential and parallel
# evaluation. These parameters can be downloaded by clicking :download:`here
# <../implementations/vqe_parallel/RY_params.npy>`.

params = np.load("vqe_parallel/RY_params.npy")

##############################################################################
# Finally, the energies as functions of rotation angle can be given using
# :class:`~.pennylane.VQECost`.

energies = [qml.VQECost(circuit, h, devs) for h in hamiltonians]

##############################################################################
# Calculating the potential energy surface
# ----------------------------------------
#
# :class:`~.pennylane.VQECost` returns a :class:`~.pennylane.QNodeCollection` which can be
# evaluated using the input parameters to the ansatz circuit. The
# :class:`~.pennylane.QNodeCollection` can be evaluated asynchronously by passing the keyword
# argument ``parallel=True``. When ``parallel=False`` (the default behaviour), the QNodes are
# instead evaluated sequentially.
#
# We can use this feature to compare the sequential and parallel times required to calculate the
# potential energy surface. The following function calculates the surface:


def calculate_surface(parallel=True):
    s = []
    t0 = time.time()

    for i, e in enumerate(energies):
        print("Running for inter-atomic distance {} Å".format(list(data.keys())[i]))
        s.append(e(params[i], parallel=parallel))

    t1 = time.time()

    print("Evaluation time: {0:.2f} s".format(t1 - t0))
    return s, t1 - t0


print("Evaluating the potential energy surface sequentially")
surface_seq, t_seq = calculate_surface(parallel=False)

print("\nEvaluating the potential energy surface in parallel")
surface_par, t_par = calculate_surface(parallel=True)

##############################################################################
# We have seen how a :class:`~.pennylane.QNodeCollection` can be evaluated in parallel. This results
# in a speed up in processing:

print("Speed up: {0:.2f}".format(t_seq / t_par))

##############################################################################
# Can you think of other ways to combine multiple QPUs to improve the
# performance of quantum algorithms? To conclude the tutorial, let's plot the calculated
# potential energy surfaces:

plt.plot(surface_seq, linewidth=2.2, marker="o", color='red')
plt.plot(surface_par, linewidth=2.2, marker="d", color='blue')
plt.title("Potential energy surface for molecular hydrogen", fontsize=12)
plt.xlabel("Atomic separation (Å)", fontsize=16)
plt.ylabel("Ground state energy (Ha)", fontsize=16)
plt.grid(True)

##############################################################################
# These surfaces overlap, with any variation due to the limited number of shots used to evaluate the
# expectation values in the ``forest.qvm`` device (we are using the default value of
# ``shots=1024``).
