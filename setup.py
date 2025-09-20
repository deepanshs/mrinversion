"""Setup for mrinversion package."""
from os.path import abspath
from os.path import dirname
from os.path import join
from setuptools import find_packages, setup


with open("mrinversion/__init__.py", encoding="utf-8") as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            version = after_keyword.strip()[1:-1]

module_dir = dirname(abspath(__file__))

install_requires = [
    "numpy>=2.0",
    "setuptools>=27.3",
    "csdmpy>=0.7",
    "mrsimulator>=1.0.0",
    "scikit-learn>=1.5.2",
    "numba>=0.61.2",
]

setup_requires = ["setuptools>=27.3", "numpy"]
extras = {"matplotlib": ["matplotlib>=3.0"]}


setup(
    name="mrinversion",
    version=version,
    description=(
        "Python based statistical learning of NMR tensor and relaxation parameters "
        "distribution."
    ),
    long_description=open(join(module_dir, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Deepansh J. Srivastava",
    author_email="deepansh2012@gmail.com",
    python_requires=">=3.10",
    url="https://github.com/DeepanshS/mrinversion/",
    packages=find_packages(),
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
