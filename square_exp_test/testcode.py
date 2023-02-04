import unittest
from main import Variable
from square_func import square
import numpy as np

class Square_test(unittest.TestCase):
    def test_square(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)