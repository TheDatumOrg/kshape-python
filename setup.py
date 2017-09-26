#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = "1.0.1"

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.2",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="kshape",
    version=__version__,
    description="Python implementation of k-Shape",
    classifiers=CLASSIFIERS,
    author="J. Thalheim",
    author_email="joerg@thalheim.io",
    packages=find_packages(),
    zip_safe=True,
    test_suite="kshape.test",
    license="",
    url="https://github.com/Mic92/kshape",
    entry_points={},
    install_requires=['numpy']
)
