r"""
QJIT compilation with Qrack and Catalyst
========================================

.. meta::
    :property="og:description": Using the Qrack device for PennyLane and Catalyst, with GPU-acceleration and novel optimization.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/qrack/qrack_logo.jpg

*Author: Dan Strano — Posted: 26 June 2024.*

How Qrack Works
---------------

`Qrack <https://github.com/unitaryfund/qrack>`__ is a GPU-accelerated quantum computer simulator with many novel optimizations, and `PyQrack <https://github.com/unitaryfund/pyqrack>`__ is its Python wrapper, written in pure (``ctypes``) Python language standard. Founded in 2017 by Dan Strano and Benn Bollay, Qrack's vision was always to provide the best possible (classical) quantum computer "emulator," targeting the use case of running industrially-relevant quantum "workloads" without recourse to genuine quantum computer hardware. Ben and Dan wanted to provide the emulator as open source, free of charge, "agnostic" to any specific GPU or hardware accelerator provider, "backwards compatible" to serve those with very limited classical computer hardware resources, (nonetheless) capable of scaling to supercomputer systems, as secure and free of external dependencies as possible, and under the reasonably permissive LGPL license, with bindings and wrappers for third-party libraries provided under even more permissive licenses like MIT and Apache 2.0. The Qrack team's hope was that the global "floor" of minimal access to cost-competitive quantum workload throughput could never be lower than the capabilities of Qrack.

When simulating quantum subroutines of varying qubit widths, Qrack will "transparently," automatically, and dynamically transition between GPU-based and CPU-based simulation techniques for maximum execution speed, when qubit registers might be too narrow to benefit from the large parallel processing element count of a GPU, up to maybe roughly 20 qubits, depending upon the classical hardware platform. Qrack also offers "hybrid" stabilizer simulation (with fallback to universal simulation) and near-Clifford simulation with greatly reduced memory footprint on Clifford gate sets with the inclusion of the `RZ` variational Pauli Z-axis rotation gate. (For more information, see the `report <https://arxiv.org/abs/2304.14969>`__ by the Qrack and Unitary Fund teams to QCE'23.)

Particularly for systems that don't rely on GPU-acceleration, Qrack offers a "quantum binary decision diagram" ("QBDD") simulation algorithm option that might significantly reduce memory footprint or execution complexity for circuits with "low entanglement," judged by the complexity of a QBDD to represent the state. (Qrack's implementation of QBDD is entirely original source code, but it based on reports like `https://arxiv.org/abs/2302.04687 <https://arxiv.org/abs/2302.04687>`__.) Qrack also offers approximation options aimed at trading off minimal fidelity reduction for maximum reduction in simulation complexity (as opposed to models of physical noise) such as "Schmidt decomposition rounding parameter" ("SDRP") and "near-Clifford rounding parameter" ("NCRP"). ("SDRP" is covered in Qrack's `report <https://arxiv.org/abs/2304.14969>`__ to QCE`23, as well.)

As you might guess from the last paragraph, the Qrack simulator doesn't fit neatly into a single canonical category of quantum computer simulation algorithm: it optionally and by default leverages elements of state vector simulation, tensor network simulation, stabilizer and near-Clifford simulation, and QBDD simulation, often all at once, while it introduces some novel algorithmic "tricks" for Schmidt decomposition of state vectors in a manner similar to "matrix product state" ("MPS") simulation.

In this tutorial you will learn how to use the Qrack device back end for PennyLane and quantum just-in-time (QJIT) compilation via Catalyst, and you'll learn certain suggested cases of use where Qrack might particularly excel at delivering lightning-fast performance or minimizing required memory resources (though Qrack is a "general-purpose" simulator, and users might employ it for all their applications and still see parity with or improvement over available device back ends).

.. figure:: ../_static/demonstration_assets/qrack/qrack_logo.jpg
    :align: center
    :width: 60%
    :target: javascript:void(0);

Demonstrating Qrack with the Quantum Fourier Transform
------------------------------------------------------

The :doc:`quantum Fourier transform (QFT) <tutorial_qft>` is a building block subroutine of many other quantum algorithms. Qrack exhibits unique capability on many cases of the QFT algorithm, and worst-case performance is competitive with other popular quantum computer simulators (as `reported <https://arxiv.org/abs/2304.14969>`__ in 2023 at IEEE QCE).

In the case of a "trivial" computation basis eigenstate input, Qrack can simulate basically as wide of a QFT as any for which you'd ask. Below, we pick a random eigenstate initialization and perform the QFT across a width of 60 qubits.
"""

import pennylane as qml
from pennylane import numpy as np
from catalyst import qjit

import matplotlib.pyplot as plt

import random

qubits = 60
dev = qml.device("qrack.simulator", qubits, shots=8)


@qjit
@qml.qnode(dev)
def circuit():
    for i in range(qubits):
        if random.uniform(0, 1) < 0.5:
            qml.X(wires=[i])
    qml.QFT(wires=range(qubits))
    return qml.sample(wires=range(qubits))


def counts_from_samples(samples):
    counts = {}
    for sample in samples:
        s = 0

        for bit in sample:
            s = (s << 1) | bit

        s = str(s)

        if s in counts:
            counts[s] = counts[s] + 1
        else:
            counts[s] = 1

    return counts


counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"QFT on {qubits} Qubits with Random Eigenstate Init. (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig1.png
#     :align: center
#     :width: 100%
#     :target: javascript:void(0);

##############################################################################
# In this image we have represented only 8 measurement samples so we can visualize it easily.
#
# This becomes harder is we request a "non-trivial" initialization. In general, Qrack will use Schmidt decomposition techniques to try to break up circuits in separable subsystems of qubits to simulate semi-independently, combining them just-in-time with Kronecker products when they need to interact, according the user's circuit definition.
#
# The circuit becomes much harder for Qrack if we randomly initialize the input qubits with Haar-random `U3 gates <https://docs.pennylane.ai/en/stable/code/api/pennylane.U3.html>`__, but the performance is still significantly better than the worst case (of `GHZ state <https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state>`__ initialization).


qubits = 12
dev = qml.device("qrack.simulator", qubits, shots=8)


@qjit
@qml.qnode(dev)
def circuit():
    for i in range(qubits):
        th = random.uniform(0, np.pi)
        ph = random.uniform(0, np.pi)
        dl = random.uniform(0, np.pi)
        qml.U3(th, ph, dl, wires=[i])
    qml.QFT(wires=range(qubits))
    return qml.sample(wires=range(qubits))


counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"QFT on {qubits} Qubits with Random U3 Init. (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig2.png
#     :align: center
#     :width: 100%
#     :target: javascript:void(0);

##############################################################################
# Alternate Simulation Algorithms (QBDD and Near-Clifford)
# --------------------------------------------------------
# By default, Qrack relies on a combination of state vector simulation, "hybrid" stabilizer and near-Clifford simulation, and Schmidt decomposition optimization techniques. Alternatively, we could use pure stabilizer simulation or QBDD simulation if the circuit is at all amenable to optimization this way.
#
# To demonstrate this, we prepare a 60 qubit GHZ state, which would commonly be intractable with state vector simulation.

qubits = 60
dev = qml.device(
    "qrack.simulator",
    qubits,
    shots=8,
    isBinaryDecisionTree=False,
    isStabilizerHybrid=True,
    isSchmidtDecompose=False,
)


@qjit
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    for i in range(1, qubits):
        qml.CNOT(wires=[i - 1, i])
    return qml.sample(wires=range(qubits))


counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"{qubits}-Qubit GHZ preparation (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig3.png
#     :align: center
#     :width: 100%
#     :target: javascript:void(0);

##############################################################################
# As you can see, Qrack was able to construct the 60-qubit GHZ state (without exceeding memory limitations), and the probability is peaked at bit strings of all 0 and all 1.
#
# It's trivial for Qrack to perform large GHZ state preparations with "hybrid" stabilizer or near-Clifford simulation, if Schmidt decomposition is deactivated.
# QBDD cannot be accelerated by GPU, so its application might be limited, but it is parallel over CPU processing elements, hence it might be particularly well-suited for systems with no GPU at all, though Qrack default simulation methods will likely still outperform on it "BQP-complete" problems like random circuit sampling or quantum volume certification.
#

qubits = 24
dev = qml.device(
    "qrack.simulator", qubits, shots=8, isBinaryDecisionTree=True, isStabilizerHybrid=False
)


@qjit
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    for i in range(1, qubits):
        qml.CNOT(wires=[i - 1, i])
    return qml.sample(wires=range(qubits))


counts = counts_from_samples(circuit())

plt.bar(counts.keys(), counts.values())
plt.title(f"{qubits}-Qubit GHZ preparation (8 samples)")
plt.xlabel("|x⟩")
plt.ylabel("counts")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig4.png
#     :align: center
#     :width: 100%
#     :target: javascript:void(0);

##############################################################################
# If your gate set is restricted to Clifford with general :func:`~.RZ` gates (being mindful of the fact that compilers like Catalyst might optimize such a gate set basis into different gates), the time complexity for measurement samples becomes doubly-exponential with near-Clifford simulation, but the space complexity is almost exactly that of stabilizer simulation for the logical qubits plus an ancillary qubit per (non-optimized) ``RZ`` gate, scaling like the square of the logical plus ancillary qubit count.
#
# Comparing performance
# ---------------------
# We've already seen, the Qrack device back end can do some tasks that most other simulators, or basically any other simulator, simply can't do, like 60-qubit-wide special cases of the QFT or GHZ state preparation with a Clifford or universal (QBDD) simulation algorithm, for example. However, literally most circuits in the space of all random ("BQP-complete") circuits will tend to be limited to the equivalent performance of state vector simulation, in practice, even for Qrack.
#
# How does Qrack compare for performance with other simulators on a "non-trivial" problem, like the U3 initialization we used above for the QFT algorithm?


import time


def bench(n, results):
    for device in ["qrack.simulator", "lightning.qubit"]:
        dev = qml.device(device, n, shots=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            for i in range(n):
                th = random.uniform(0, np.pi)
                ph = random.uniform(0, np.pi)
                dl = random.uniform(0, np.pi)
                qml.U3(th, ph, dl, wires=[i])
            qml.QFT(wires=range(n))
            return qml.sample(wires=range(n))

        start_ns = time.perf_counter_ns()
        circuit()
        results[
            f"Qrack ({n} qb)" if device == "qrack.simulator" else f"Lightning ({n} qb)"
        ] = time.perf_counter_ns() - start_ns

    return results


results = {}
results = bench(6, results)
results = bench(12, results)
results = bench(18, results)

bar_colors = ["purple", "yellow", "purple", "yellow"]
plt.bar(results.keys(), results.values(), color=bar_colors)
plt.title("Performance comparison, QFT with U3 initialization (1 sample apiece)")
plt.xlabel("|x⟩")
plt.ylabel("Nanoseconds")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig5.png
#     :align: center
#     :width: 100%
#     :target: javascript:void(0);

##############################################################################
# Benchmarks will differ somewhat when running this code on your local machine, for example, but we tend to see that Qrack manages to demonstrate good performance compared to the `Lightning simulators <https://docs.pennylane.ai/projects/lightning/en/stable/index.html>`__ on this task case. (Note that this initialization case isn't specifically the hardest case of the QFT for Qrack; that's probably rather a GHZ state input.)
#
# Similarly, we're using quantum just-in-time compilation from Catalyst, for both Qrack and Lightning. How does Qrack with QJIT compare to Qrack without it?

def bench(n, results):
    dev = qml.device("qrack.simulator", n, shots=1)

    @qjit
    @qml.qnode(dev)
    def circuit():
        for i in range(n):
            th = random.uniform(0, np.pi)
            ph = random.uniform(0, np.pi)
            dl = random.uniform(0, np.pi)
            qml.U3(th, ph, dl, wires=[i])
        qml.QFT(wires=range(n))
        return qml.sample(wires=range(n))

    start_ns = time.perf_counter_ns()
    circuit()
    results[f"QJIT Qrack ({n} qb)"] = time.perf_counter_ns() - start_ns

    @qml.qnode(dev)
    def circuit():
        for i in range(n):
            th = random.uniform(0, np.pi)
            ph = random.uniform(0, np.pi)
            dl = random.uniform(0, np.pi)
            qml.U3(th, ph, dl, wires=[i])
        qml.QFT(wires=range(n))
        return qml.sample(wires=range(n))

    start_ns = time.perf_counter_ns()
    circuit()
    results[f"PyQrack ({n} qb)"] = time.perf_counter_ns() - start_ns

    return results

# Make sure OpenCL has been initalized in PyQrack:
bench(6, results)

results = {}
results = bench(6, results)
results = bench(12, results)
results = bench(18, results)

bar_colors = ["purple", "yellow", "purple", "yellow"]
plt.bar(results.keys(), results.values(), color=bar_colors)
plt.title("Performance comparison, QFT with U3 initialization (1 sample apiece)")
plt.xlabel("|x⟩")
plt.ylabel("Nanoseconds")
plt.show()

##############################################################################
# .. figure:: ../_static/demonstration_assets/qrack/fig6.png
#     :align: center
#     :width: 100%
#     :target: javascript:void(0);

##############################################################################
# Again, "your mileage may vary" somewhat, depending on your local system, but Qrack tends to be significantly faster with Catalyst QJIT than without!
#
# As a basic test of validity, if we compare the inner product between both simulator state vector outputs on some QFT case, do they agree?


def validate(n):
    results = []
    for device in ["qrack.simulator", "lightning.qubit"]:
        dev = qml.device(device, n, shots=1)

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            for i in range(1, n):
                qml.CNOT(wires=[i - 1, i])
            qml.QFT(wires=range(n))
            return qml.state()

        start_ns = time.perf_counter_ns()
        results.append(circuit())

    return np.abs(sum([np.conj(x) * y for x, y in zip(results[0], results[1])]))


print("Qrack cross entropy with Lightning:", validate(12), "out of 1.0")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  .. code-block:: none
#
#       Qrack cross entropy with Lightning: 0.9999997797266185 out of 1.0

##############################################################################
# Conclusion
# ----------
# In this tutorial, we've demonstrated the basics of using the Qrack simulator back end and showed readers examples of special cases on which Qrack's novel optimizations can lead to huge increases in performance or maximum achievable qubit widths. Remember the Qrack device back end for PennyLane if you'd like to leverage GPU acceleration but don't want to complicate your choice of devices or device initialization, to handle a mixture of wide and narrow qubit registers in your subroutines.
#
# About the author
# ----------------
# .. include:: ../_static/authors/dan_strano.txt
