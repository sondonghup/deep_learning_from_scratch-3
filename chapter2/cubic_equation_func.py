'''
3차 방정식
'''
import numpy as np
from main import Function

class Cubic_equation(Function):
    def equation(self, x: np.ndarray) -> float:
        return 3 * (x ** 3) - 2 * (x ** 2) + 5 

    def forward(self, input: np.ndarray) -> float:
        return self.equation(input)

    def backward(self, gy: np.ndarray) -> float:
        x = self.input.data
        gx = ( 9 * (x ** 2) - 4 * x ) * gy
        return gx

def cubic_equation(x):
    return Cubic_equation()(x)