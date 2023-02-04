from main import Function
import numpy as np

class Exp(Function):
    def forward(self, input):
        return np.exp(input)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx