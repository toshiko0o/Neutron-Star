from setuptools import setup
from Cython.Build import cythonize

# cython: language_level=3

setup(
    ext_modules=cythonize("simple_cfunc.pyx"),
)