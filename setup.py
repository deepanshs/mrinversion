# import platform
from os.path import abspath
from os.path import dirname
from os.path import join

from setuptools import setup

# from os.path import split
# import numpy as np
# import numpy.distutils.system_info as sysinfo
# from Cython.Build import cythonize
# from setuptools import Extension


with open("mrinversion/__init__.py", "r") as f:
    for line in f.readlines():
        if "__version__" in line:
            before_keyword, keyword, after_keyword = line.partition("=")
            version = after_keyword.strip()[1:-1]

module_dir = dirname(abspath(__file__))

# include_dirs = [
#     "/opt/local/include/",
#     "/usr/include/",
#     "/usr/include/openblas",
#     "/usr/include/x86_64-linux-gnu/",
# ]

# library_dirs = [
#     "/opt/local/lib/",
#     "/usr/lib64/",
#     "/usr/lib/",
#     "/usr/lib/x86_64-linux-gnu/",
# ]

# libraries = []
# data_files = []

# numpy_include = np.get_include()

# if platform.system() == "Windows":
#     conda_location = numpy_include
#     for _ in range(5):
#         conda_location = split(conda_location)[0]
#     include_dirs += [join(conda_location, "Library", "include", "openblas")]
#     include_dirs += [join(conda_location, "Library", "include")]
#     include_dirs += [join(conda_location, "include")]
#     library_dirs += [join(conda_location, "Library", "lib")]
#     libraries += ["openblas"]
#     name = "openblas"

#     extra_link_args = ["-lm"]
#     extra_compile_args = []

# # this section is important for travis-ci build.
# else:
#     libraries = ["openblas", "pthread"]
#     openblas_info = sysinfo.get_info("openblas")

#     if openblas_info != {}:
#         name = "openblas"
#         library_dirs += openblas_info["library_dirs"]
#         libraries += openblas_info["libraries"]

#     extra_link_args = ["-lm"]
#     extra_compile_args = ["-g", "-O3"]

# include_dirs = list(set(include_dirs))
# library_dirs = list(set(library_dirs))
# libraries = list(set(libraries))

# include_dirs += ["mrinversion/c_lib/include/", numpy_include]

# print(include_dirs)
# print(library_dirs)
# print(libraries)

# print(extra_compile_args)
# print(extra_link_args)

# ext_modules = [
#     Extension(
#         name="mrinversion.minimizer.fista",
#         sources=[
#             "mrinversion/c_lib/src/fista_GD.c",
#             "mrinversion/c_lib/src/fista.pyx",
#         ],
#         include_dirs=include_dirs,
#         language="c",
#         libraries=libraries,
#         library_dirs=library_dirs,
#         data_files=data_files,
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args,
#     )
# ]


# ext_modules = [
#     Extension(
#         name=NAME + ".minimizer.fista",
#         sources=_source_files,
#         # include_dirs=[numpy.get_include()],
#         # extra_objects= ['NMRlib'], # ["fc.o"],  # if you compile fc.cpp separately
#         include_dirs=include_lib_directories,  # .../site-packages/numpy/core/include
#         language="c",
#         # libraries=['iomp5', 'pthread'],
#         extra_compile_args="-O1 -flax-vector-conversions -g ".split(),
#         extra_link_args="-g -lmkl_intel_lp64 -lmkl_intel_thread \
#                         -lmkl_core -ldl -liomp5 -lm -W".split()
#         # extra_link_args = "-g".split()
#     )
# ]

setup(
    name="mrinversion",
    version=version,
    description="A python program for inverting NMR anisotropic lineshapes",
    long_description=open(join(module_dir, "README.md")).read(),
    author="Deepansh J. Srivastava",
    author_email="deepansh2012@gmail.com",
    python_requires=">=3.6",
    url="https://github.com/DeepanshS/mrinversion/",
    # packages=find_packages("src"),
    # package_dir={"": "mrinversion"},
    setup_requires=["numpy>=1.13.3", "setuptools>=27.3", "cython>=0.29.11"],
    install_requires=[
        "numpy>=1.13.3",
        "setuptools>=27.3",
        "cython>=0.29.11",
        "matplotlib>=3.0.2",
        "csdmpy>=0.1.4",
        "mrsimulator>=0.2.0",
    ],
    extras_require={"fancy feature": ["plotly>=3.6", "dash>=0.40", "dash_daq>=0.1"]},
    # ext_modules=cythonize(ext_modules, language_level=3),
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


# setup(
#     name=NAME,
#     version=about["__version__"],
#     description=DESCRIPTION,
#     long_description=long_description,
#     author=AUTHOR,
#     author_email=EMAIL,
#     python_requires=REQUIRES_PYTHON,
#     url=URL,
#     packages=find_packages(),
#     install_requires=REQUIRED,
#     extras_require=EXTRAS,
#     cmdclass=cmdclass,
#     ext_modules=ext,
#     classifiers=[
#         # Trove classifiers
#         # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
#         "License :: OSI Approved :: BSD License",
#         "Programming Language :: Python :: 3",
#     ],
# )
