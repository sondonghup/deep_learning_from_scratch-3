import unittest
from main import Variable
from square_func import square
from exp_func import exp
from cubic_equation_func import cubic_equation
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--func', default='cubic')
'''
unittest가 있어서 testcode의 인자가 자꾸 unittest로 들어가는것 같다 
-m unittest testcode.py -f square 으로 했을때 오류가 나서
testcode.py에 unittest.main()을 넣은뒤 testcode.py -f square으로 해도 오류가 난다
'''
args = parser.parse_args()

print(args.func)

def diffrentianl(f, x, eps = 1e-4):
    print(type(x))
    print(x.data)
    x1 = Variable(np.array(x.data - eps))
    x2 = Variable(np.array(x.data + eps))
    y1 = f(x1)
    y2 = f(x2)
    result = (y2.data - y1.data) / (2 * eps)
    return result

if args.func == 'square':
    print('square 입니다.')
    class Square_test(unittest.TestCase):
        def test_square(self):
            x = Variable(np.array(2.0))
            y = square(x)
            expected = np.array(4.0)
            self.assertEqual(y.data, expected)
        
        def test_square_backward(self):
            x = Variable(np.array(3.0))
            y = square(x)
            y.backward()
            expected = np.array(7.0)
            self.assertEqual(x.grad, expected)

        def test_auto_square_backward(self):
            x = Variable(np.array(3.0))
            y = square(x)
            y.backward()
            expected = diffrentianl(square, x)
            self.assertAlmostEqual(x.grad, expected)
            # compare = np.allclose(x.grad, expected)
            # self.assertTrue(compare)

elif args.func == 'exp':
    print('exp 입니다.')
    class Exp_test(unittest.TestCase):
        def test_exp(self):
            x = Variable(np.array(2.0))
            y = exp(x)
            expected = np.array(np.exp(np.array(2.0)))
            self.assertEqual(y.data, expected)

        def test_exp_backward(self):
            x = Variable(np.array(2.0))
            y = exp(x)
            expected = np.array(np.exp(np.array(2.0)))
            self.assertEqual(y.data, expected)

        def test_auto_exp_backward(self):
            x = Variable(np.array(3.0))
            y = exp(x)
            y.backward()
            expected = diffrentianl(exp, x)
            self.assertAlmostEqual(x.grad, expected)
            # compare = np.allclose(x.grad, expected)
            # self.assertTrue(compare)

elif args.func == 'cubic':
    print('cubic equation 입니다.')
    class Cubic_equation_test(unittest.TestCase):
        def test_cubic_equation(self):
            x = Variable(np.array(1.0))
            y = cubic_equation(x)
            expected = np.array(4.0)
            self.assertEqual(y.data, expected)
    
        def test_cubic_equation_backward(self):
            x = Variable(np.array(1.0))
            y = cubic_equation(x)
            y.backward()
            expected = np.array(3.0)
            self.assertAlmostEqual(x.grad, expected)
        
        def test_auto_cubic_equation_backward(self):
            x = Variable(np.array(1.0))
            y = cubic_equation(x)
            y.backward()
            expected = diffrentianl(cubic_equation, x)
            self.assertAlmostEqual(x.grad, expected)

unittest.main()