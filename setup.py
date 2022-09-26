from setuptools import setup, find_packages

__version__ = "1.0.3"

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="kshape",
    version=__version__,
    description="k-Shape",
    classifiers=CLASSIFIERS,
    author="Teja",
    author_email="tejabogireddy19@gmail.com",
    packages=find_packages(),
    zip_safe=True,
    license="",
    url="https://github.com/TheDatumOrg/kshape-python",
    entry_points={},
    install_requires=['numpy']
)
