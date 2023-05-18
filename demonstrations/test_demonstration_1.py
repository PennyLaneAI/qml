r"""

Formatting Test Demonstration
=============================

.. meta::
    :property="og:description": A demonstration for testing different formatting features.

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Here we have some *italic text* and some **bold text**.

A bulleted list

* List Item 1
* List Item 2
* List Item 3

A numbered list

1. List Item 1
2. List Item 2
3. List Item 3

Another numbered list

#. List Item 1
#. List Item 2
#. List Item 3

A Heading
---------

Here we have an inline ``code block``.

Below are some code blocks that are *within* an RST comment, so are not processed as code.

.. code-block:: bash

    pip install amazon-braket-pennylane-plugin

.. code-block:: none

    Execution time on remote device (seconds): 3.5898206680030853
    Execution time on local device (seconds): 23.50668462700196

Figures
-------

Centre-aligned

.. figure:: ../_static/remote-multi-job-simulator.png
    :align: center
    :scale: 75%
    :alt: PennyLane can leverage Braket for parallelized gradient calculations

Left-aligned

.. figure:: ../_static/remote-multi-job-simulator.png
    :align: left
    :scale: 75%
    :alt: PennyLane can leverage Braket for parallelized gradient calculations

Right-aligned

.. figure:: ../_static/remote-multi-job-simulator.png
    :align: right
    :scale: 75%
    :alt: PennyLane can leverage Braket for parallelized gradient calculations

"""

##############################################################################
# .. important::
#
#     This is an 'important' block. It should have an orange background.
#
#     * List Item 1
#     * List Item 2
#     * List Item 3
#
#     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
#
# .. admonition:: Definition
#     :class: defn
#
#     This is a 'definition' block. It should have no background colour, a green title, and padding around it that makes the text narrower than that above and below.
#
#     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
#
# .. tip::
#
#     This is a 'tip' block. It should have a blue background.
#
#     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
#
# .. note::
#
#     This is a 'note' block. It should have a green background.
#
#     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
#
# .. warning::
#
#     This is a 'warning' block. It should have an orange background.
#
#     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
#
# Some 'includes'
# -----------------
# .. include:: ../_static/authors/thomas_bromley.txt
#
# .. include:: ../_static/authors/maria_schuld.txt