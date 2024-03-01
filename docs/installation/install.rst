============
Installation
============

Requirements
------------

``mrinversion`` has the following strict requirements:

- `Python <https://www.python.org>`_ 3.8 or later
- `Numpy <https://numpy.org>`_ 1.20 or later

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

For *Mac* users, python version 3 may be installed under the name *python3*. You may replace
*python* for *python3* in the above command and all subsequent python statements.


Installing ``mrinversion``
--------------------------

.. only:: html

  .. tabs::

    .. tab:: Google Colab Notebook

      .. include:: colab.rst

    .. tab:: Local machine (Using pip)

      .. include:: pip.rst

.. only:: not html

  Google Colab Notebook
  '''''''''''''''''''''
  .. include:: colab.rst

  Local machine (Using pip)
  '''''''''''''''''''''''''
  .. include:: pip.rst


Upgrading to a newer version
""""""""""""""""""""""""""""

To upgrade, type the following in the terminal/Prompt,

.. code-block:: bash

    $ pip install mrinversion -U
