import doctest
import unittest

from kshape import core

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(core))
    return tests
