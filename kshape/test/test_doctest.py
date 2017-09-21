import doctest
import unittest

from kshape import core

class DocTests(unittest.TestCase):
  def test_core(self):
      testSuite = unittest.TestSuite()
      testSuite.addTest(doctest.DocTestSuite(core))
      unittest.TextTestRunner(verbosity = 2).run(testSuite)
