'''
exp 함수
'''

from main import Function
import numpy as np

class Exp(Function):
    def forward(self, input: np.ndarray) -> float:
        return np.exp(input)

    def backward(self, gy:np.ndarray) -> float:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x): # 함수의 사용을 용이하게 만들기 위해
    return Exp()(x)