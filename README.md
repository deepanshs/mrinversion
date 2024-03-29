# Mrinversion

|              |                                                                                                                                                                                                                                                                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deployment   | [![PyPI version](https://img.shields.io/pypi/v/mrinversion.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/mrinversion) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mrinversion)                                                                                                                                    |
| Build Status | [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/deepanshs/mrinversion/continuous-integration-pip.yml?logo=GitHub)](https://github.com/DeepanshS/mrinversion/actions/workflow/continuous-integration-pip.yml) [![Read the Docs](https://img.shields.io/readthedocs/mrinversion)](https://mrinversion.readthedocs.io/en/latest/) |
| License      | [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)                                                                                                                                                                                                                                       |
| Metrics      | [![codecov](https://codecov.io/gh/DeepanshS/mrinversion/branch/master/graph/badge.svg)](https://codecov.io/gh/DeepanshS/mrinversion)                                                                                                                                                                                                                            |

The `mrinversion` python package is based on the statistical learning technique for determining the underlying distribution of the magnetic resonance (NMR) parameters.

The library utilizes the [mrsimulator](https://mrsimulator.readthedocs.io/en/latest/)
package for generating solid-state NMR spectra and
[scikit-learn](https://scikit-learn.org/latest/) package for statistical learning.

---

## Features

The `mrinversion` package includes

- **Spectral Inversion**: Two-dimensional solid-state NMR spectrum of dilute spin-systems correlating
  the isotropic to anisotropic frequencies to a three-dimensional distribution of NMR tensor parameters.
  Presently, we support the inversion of

  - **Magic angle turning (MAT), Phase adjusted spinning sidebands (PASS)**, and similar
    spectra correlating the isotropic chemical shift resonances to pure anisotropic
    spinning sideband resonances into a three-dimensional distribution of
    nuclear shielding tensor parameters---isotropic chemical shift, shielding
    anisotropy and asymmetry parameters---defined using the Haeberlen convention.

  - **Magic angle flipping (MAF)** spectra correlating the isotropic chemical shift
    resonances to pure anisotropic resonances into a three-dimensional distribution of
    nuclear shielding tensor parameters---isotropic chemical shift, shielding
    anisotropy and asymmetry parameters---defined using the Haeberlen convention.

- **Relaxometry Inversion**: Inversion of NMR relaxometry measurements to the distribution of
  relaxation parameters (T1, T2).

For more information, refer to the
[documentation](https://mrinversion.readthedocs.io/en/latest/).

> **View our example gallery**
>
> [![](https://img.shields.io/badge/View-Example%20Gallery-Purple?s=small)](https://mrinversion.readthedocs.io/en/latest/galley_examples/index.html)

## Installation

    $ pip install mrinversion

Please read our [installation document](https://mrinversion.readthedocs.io/en/latest/installation.html) for details.

## How to cite

If you use this work in your publication, please cite the following.

- Srivastava, D. J.; Grandinetti P. J., Statistical learning of NMR tensors from 2D
  isotropic/anisotropic correlation nuclear magnetic resonance spectra, J. Chem. Phys.
  **153**, 134201 (2020). [DOI:10.1063/5.0023345](https://doi.org/10.1063/5.0023345).

- Deepansh J. Srivastava, Maxwell Venetos, Philip J. Grandinetti, Shyam Dwaraknath, & Alexis McCarthy. (2021, May 26). mrsimulator: v0.6.0 (Version v0.6.0). Zenodo. http://doi.org/10.5281/zenodo.4814638

Additionally, if you use the CSDM data model, please consider citing

- Srivastava DJ, Vosegaard T, Massiot D, Grandinetti PJ (2020) Core Scientific Dataset Model: A lightweight and portable model and file format for multi-dimensional scientific data. PLOS ONE 15(1): e0225953. https://doi.org/10.1371/journal.pone.0225953
