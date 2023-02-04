'''
3차 방정식
'''
import numpy as np
from main import Function

class Cubic_equation(Function):
    def equation(self, x: np.ndarray) -> float:
        return 3 * (x ** 3) - 2 * (x ** 2) + 5 

    def forward(self, inputs: np.ndarray) -> float:
        outputs = []
        for input in inputs:
            outputs.append(self.equation(input))
        return outputs

    def backward(self, gys: np.ndarray) -> float:
        x = self.input.data
        gxs = []
        for x, gy in zip(self.input.data, gys):
            gx = ( 9 * (x ** 2) - 4 * x ) * gy
            gxs.append(gx)
        return gxs 

def cubic_equation(x):
    print('시발련아', x)
    return Cubic_equation()(x)