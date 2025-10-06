import numpy as np
import unittest
from functions.exercise1 import KNN

# Unit Tests on basic datasets
class TestKNN(unittest.TestCase):

    def setUp(self):
        self.x_train = np.array([[0, 0], [1, 1], [2, 2], [10, 10]])
        self.class_train = np.array([0, 0, 1, 1])

    def test_single_point_k1(self):
        x_test = np.array([[0.1, 0.1]])
        pred = KNN(self.x_train, self.class_train, x_test, N_test=1, K=1)
        self.assertEqual(pred[0], 0)

    def test_single_point_k3(self):
        x_test = np.array([[1.5, 1.5]])
        pred = KNN(self.x_train, self.class_train, x_test, N_test=1, K=3)
        self.assertEqual(pred[0], 0)

    def test_multiple_points(self):
        x_test = np.array([[0.1, 0.1], [9, 9]])
        pred = KNN(self.x_train, self.class_train, x_test, N_test=2, K=1)
        np.testing.assert_array_equal(pred, np.array([0, 1]))
