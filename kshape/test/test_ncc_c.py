import numpy as np
import unittest

from kshape.core import _ncc_c, _ncc_c_2dim, _ncc_c_3dim, zscore


class NccTest(unittest.TestCase):
  def test_ncc_c_2_and_3dim_matches(self):
      test = zscore(np.array([
          [1, 2, 3, 4, 5],
          [0, 10, 4, 5, 7],
          [-1, 15, -12, 8, 9],
      ]), axis=1)
      centroids = np.array([[1, 1, 0, 1, 1], [10, 12, 0, 0, 1]])
      distances1 = np.empty((3, 2))
      distances2 = np.empty((3, 2))
      for i in range(3):
          for j in range(2):
              distances1[i, j] = 1 - _ncc_c(test[i], centroids[j]).max()

      for j in range(2):
          distances2[:,j] = 1 - _ncc_c_2dim(test, centroids[j]).max(axis=1)

      distances3 = (1 - _ncc_c_3dim(test, centroids).max(axis=2)).T

      np.testing.assert_array_equal(distances1, distances2)
      np.testing.assert_array_equal(distances2, distances3)
