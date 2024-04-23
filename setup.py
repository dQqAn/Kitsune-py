from Cython.Build import cythonize
from setuptools import setup
#https://www.geeksforgeeks.org/differences-between-distribute-distutils-setuptools-in-python/

setup(
    ext_modules=cythonize(["*.pyx"])
)
