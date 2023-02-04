import numpy as np
from utils import as_array

class Variable:
    def __init__(self, data):
        if data is not None:
            # data = as_array(data)
            if not isinstance(data, np.ndarray):
                raise TypeError('np.ndarray 자료형을 입력으로 넣어주세요')

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        funcs = [self.creator]

        # if self.grad == None:
        #     self.grad = np.array(1.0)

        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            if y.grad == None:
                # y.grad = np.array(1.0)
                y.grad = np.ones_like(self.data)
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function():
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        
        return output

    def forward(self, x):
        raise NotImplementedError('순전파')
    
    def backward(self, gy):
        raise NotImplementedError('역전파')