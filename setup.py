# -*- coding: utf-8 -*-
from os.path import abspath
from os.path import dirname
from os.path import join

from setuptools import setup

with open("mrinversion/__init__.py", "r") as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            version = after_keyword.strip()[1:-1]

module_dir = dirname(abspath(__file__))

setup(
    name="mrinversion",
    version=version,
    description=(
        "Python based statistical learning of tensor distribution from NMR "
        "anisotropic spectra."
    ),
    long_description=open(join(module_dir, "README.md")).read(),
    author="Deepansh J. Srivastava",
    author_email="deepansh2012@gmail.com",
    python_requires=">=3.6",
    url="https://github.com/DeepanshS/mrinversion/",
    # packages=find_packages("src"),
    # package_dir={"": "mrinversion"},
    setup_requires=["setuptools>=27.3"],
    install_requires=[
        "numpy>=1.17",
        "setuptools>=27.3",
        "matplotlib>=3.0.2",
        "csdmpy>=0.3",
        "mrsimulator>=0.3",
    ],
    include_package_data=True,
    zip_safe=False,
    license="BSD-3-Clause",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
