from os.path import abspath
from os.path import dirname
from os.path import join

from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
import numpy as np

with open("mrinversion/__init__.py") as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            version = after_keyword.strip()[1:-1]

module_dir = dirname(abspath(__file__))

install_requires = [
    "numpy<2.0",
    "setuptools>=27.3",
    "csdmpy>=0.6",
    "mrsimulator>=0.8.0rc0",
    "scikit-learn>=0.22",
    "pydantic<=1.10",
]

setup_requires = ["setuptools>=27.3", "numpy"]
extras = {"matplotlib": ["matplotlib>=3.0"]}

ext1 = Extension(
    name="mrinversion.linear_model.fista.fista",
    sources=["mrinversion/linear_model/fista/fista.f90"],
    include_dirs=[np.get_include()],
    # f2py_options=['only:', 'subroutine_name', ':'],
    extra_f90_compile_args=["-O3"],  # '-fopenmp'],
    libraries=["gomp", "blas"],
    # f2py_options=['only:', 'fista', ':'],
    language="f90",
)

ext2 = Extension(
    name="mrinversion.linear_model.fista.fista_cv",
    sources=["mrinversion/linear_model/fista/fista_cv.f90"],
    include_dirs=[np.get_include()],
    # f2py_options=['only:', 'subroutine_name', ':'],
    extra_f90_compile_args=["-O3"],  # '-fopenmp'],
    libraries=["gomp", "blas"],
    # f2py_options=['only:', 'fista', ':'],
    language="f90",
)

setup(
    name="mrinversion",
    version=version,
    description=(
        "Python based statistical learning of NMR tensor and relaxation parameters "
        "distribution."
    ),
    long_description=open(join(module_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="Deepansh J. Srivastava",
    author_email="deepansh2012@gmail.com",
    python_requires=">=3.8",
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
    ext_modules=[ext1, ext2],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
)
