

============
Installation
============

Requirements
------------

``mrinversion`` has the following strict requirements:

- `Python <https://www.python.org>`_ 3.6 or later
- `Numpy <https://numpy.org>`_ 1.17 or later

See :ref:`requirements` for a full list of requirements.

Make sure you have the required version of python by typing the following in the
terminal,

.. tip::
    You may also click the copy-button located at the top-right corner of the code cell
    area in the HTML docs, to copy the code lines without the prompts and then paste it
    as usual.
    Thanks to `Sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/>`_)

.. code-block:: shell

      $ python --version

For `Mac` users, python version 3 may be installed under the name `python3`. You may replace
`python` for `python3` in the above command and all subsequent python statements.

Installing ``mrinversion``
--------------------------

On Local machine (Using pip)
''''''''''''''''''''''''''''

The ``mrinversion`` package utilizes the `mrsimulator <https://mrsimulator.readthedocs.io/en/latest/>`_
package for generating the NMR line-shapes.

For **Linux** and **Mac** users, type the following in the terminal to install the
package.

.. code-block:: bash

    $ pip install mrinversion

For **Windows** users, first, `install <https://mrsimulator.readthedocs.io/en/latest/installation.html#on-local-machine-using-pip>`_
the mrsimulator package and then install the mrinversion package using the above command.

If you get a ``PermissionError``, it usually means that you do not have the required
administrative access to install new packages to your Python installation. In this
case, you may consider adding the ``--user`` option, at the end of the statement, to
install the package into your home directory. You can read more about how to do this in
the `pip documentation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

.. code-block:: bash

    $ pip install mrinversion --user

Upgrading to a newer version
""""""""""""""""""""""""""""

To upgrade, type the following in the terminal/Prompt,

.. code-block:: bash

    $ pip install mrinversion -U

On Google Colab Notebook
''''''''''''''''''''''''

Colaboratory is a Google research project. It is a Jupyter notebook environment that
runs entirely in the cloud. Launch a new notebook on
`Colab <http://colab.research.google.com>`_. To install the mrinversion package, type

.. code-block:: shell

    !pip install mrinversion

in the first cell, and execute. All done! You may now start using the library.
