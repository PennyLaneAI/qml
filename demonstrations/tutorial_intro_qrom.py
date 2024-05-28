r"""Intro to QROM
=============================================================


Intro, hook

QROM
-----

- Define QROM.
- Talk about naming of the wires since there are 3 different registers and it could be confusing.
- We will see 3 different implementations: Select, SelectSwap and an advanced version of the last one.

Select
~~~~~~~

- Introduce the Select operator and how it can be particularized to define a QROM.
- Draw circuit (code example).
- Talk about complexity of multicontrols. Here we can mention the google paper as a strategy to reduce this complexity.

Select SWAP
~~~~~~~~~~~~

- new strategy, storing bitstrings in parallel on different qubits.
- Present idea of Select Swap. (no code)

.. figure:: ../_static/demonstration_assets/intro_qrom/qrom.gif
    :align: center
    :width: 50%
    :target: javascript:void(0)

- Problem: auxiliary qubits are not reusable.

reusable Select SWAP
~~~~~~~~~~~~~~~~~~~~~

- (There is no official name for this version of select swap. This is a name I have chosen)
- Present the template of the circuit. (picture, no code)
- Mathematical intuition (somewhat elegant)
- final curious fact: aux qubits no need to be initialize to zero. It will work with any arbitrary input.

QROM in Pennylane
-----------------

- Show an example to visually see Select SWAP with the two variants


Conclusion
-----------------

"""

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/juan_miguel_arrazola.txt
