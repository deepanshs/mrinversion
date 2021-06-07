# -*- coding: utf-8 -*-
from os.path import abspath
from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup

with open("mrinversion/__init__.py", "r") as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            version = after_keyword.strip()[1:-1]

module_dir = dirname(abspath(__file__))

install_requires = [
    "numpy>=1.17",
    "setuptools>=27.3",
    "csdmpy>=0.4",
    "mrsimulator>=0.6",
    "scikit-learn>=0.22",
]

setup_requires = ["setuptools>=27.3"]
extras = {"matplotlib": ["matplotlib>=3.0"]}

setup(
    name="mrinversion",
    version=version,
    description=(
        "Python based statistical learning of NMR tensor parameters distribution "
        "from 2D isotropic/anisotropic NMR correlation spectra."
    ),
    long_description=open(join(module_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="Deepansh J. Srivastava",
    author_email="deepansh2012@gmail.com",
    python_requires=">=3.6",
    url="https://github.com/DeepanshS/mrinversion/",
    packages=find_packages(),
    # package_dir={"": "mrinversion"},
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras,
    tests_require=["pytest", "pytest-runner"],
    include_package_data=True,
    zip_safe=False,
    license="BSD-3-Clause",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
