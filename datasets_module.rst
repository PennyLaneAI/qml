.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>


Data Module in PennyLane
=========================

The `data` module provides functionality to access, store and manipulate the quantum datasets within the PennyLane framework.

Dataset Structure
------------------

PennyLane's quantum dataset currently contains two subcategories: `qchem` and `qspin`, which
contains data regarding molecules and spin systems, respectively. Users can use the 
:func:`~.pennylane.qdata.list_datasets` method to get a snapshot of the current state of the
datasets as we show below:

.. code-block:: python

    >>> from pprint import pprint
    >>> print('Level 1:'); pprint(qdata.list_datasets(), depth=1)
    Level 1:
    {'qchem': {...}, 'qspin': {...}}
    >>> print('Level 2:'); pprint(qdata.list_datasets(), depth=2)
    Level 2:
    {'qchem': {'H2': {...}, 'LiH': {...}, 'NH3': {...}, ...},
     'qspin': {'Heisenberg': {...}, 'Ising': {...}, ...}}

This nested-dictionary structure can also be used to generate arguments for the :func:`~.pennylane.qdata.load`
function that allows us to downlaod the dataset to be stored and accessed locally. The main purpose of these
arguments is to proivde users with the flexibility of filtering data as per their needs and downloading what
matches their specified criteria. For example, 

.. code-block:: python

    >>> qdata.get_params(qdata.list_datasets(), "qchem", basis=["STO3G"])
    [{'molname': ['full'], 'basis': ['STO3G'], 'bondlength': ['full']}]
    >>> qdata.get_keys("qchem", qdata.get_params(qdata.list_datasets(), "qchem",)[0])
    ['full']

These arguments can be supplied as it is with the load function or users can manually built these arguments
as per their liking. Upon doing so, they can simply load the data as follows:

.. code-block:: python

    >>> data_type = "qchem"
    >>> data_params = {"molname":"full", "basis":"full", "bondlength":"full"}
    >>> dataset = qdata.load(data_type, data_params)
    Downloading data to datasets/qchem
    [███████████████████████████████████████████████████████ 100.0 %] 146.48 KB/146.48 KB
    >>> dataset
    [<pennylane.qdata.chemdata.Chemdata at 0x1666b2c50>,
     <pennylane.qdata.chemdata.Chemdata at 0x1666b2500>,
     <pennylane.qdata.chemdata.Chemdata at 0x28917aec0>]

Using Datasets in PennyLane
----------------------------

Once downloaded and loaded to the memory, users can access various attributes from each of these datasets. These
attributes can then be used within the usual PennyLane workflow. For example, using the dataset downloaded above

.. code-block:: python

    >>> qchem_dataset = dataset[0]
    >>> {"mol":qchem_dataset.molecule.symbols, "ham":qchem_dataset.hamiltonian}
    {'mol: ['N', 'H', 'H', 'H'],
     'ham': <Hamiltonian: terms=4409, wires=[0, 1, 2, ... , 15]>}
    >>> dev = qml.device('lightning.qubit', wires=16, batch_obs=True)
    >>> @qml.qnode(dev, diff_method="parameter-shift")
    >>> def cost_fn_2():
    ...     qchem_dataset.adaptive_circuit(qchem_dataset.adaptive_params, range(16))
    ...     return qml.expval(qchem_dataset.num_op)
    >>> cost_fn_2()
    tensor(10., requires_grad=True)

.. toctree::
    :maxdepth: 2
    :hidden:
