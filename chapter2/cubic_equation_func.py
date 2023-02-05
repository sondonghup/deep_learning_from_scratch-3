'''
3차 방정식
'''
import numpy as np
from main import Function

class Cubic_equation(Function):
    def equation(self, x: np.ndarray) -> float:
        return 3 * (x ** 3) - 2 * (x ** 2) + 5 

    def forward(self, inputs: np.ndarray) -> float:
        return self.equation(inputs)

    def backward(self, gys: np.ndarray) -> float:
        x = self.inputs[0].data
        gxs = ( 9 * (x ** 2) - 4 * x ) * gys
        return gxs 

def cubic_equation(x):
    print('시발련아', x)
    return Cubic_equation()(x)