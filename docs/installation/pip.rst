The ``mrinversion`` package utilizes the `mrsimulator <https://mrsimulator.readthedocs.io/en/latest/>`_
package for generating the NMR line-shapes.

To install mrinversion, type the following in the terminal.

.. code-block:: bash

    $ pip install mrinversion

If you get a ``PermissionError``, it usually means that you do not have the required
administrative access to install new packages to your Python installation. In this
case, you may consider adding the ``--user`` option, at the end of the statement, to
install the package into your home directory. You can read more about how to do this in
the `pip documentation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

.. code-block:: bash

    $ pip install mrinversion --user
