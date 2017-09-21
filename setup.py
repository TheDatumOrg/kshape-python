#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = "1.0"

setup(
    name="kshape",
    version=__version__,
    description="Python implementation of kshape",
    packages=find_packages(exclude=["*.tests"]),
    entry_points={}
)
