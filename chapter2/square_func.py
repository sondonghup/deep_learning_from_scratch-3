'''
square 함수
'''

from main import Function
import numpy as np

class Square(Function):
    def forward(self, x: np.ndarray) -> float:
        return x ** 2

    def backward(self, gy: np.ndarray) -> float:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)