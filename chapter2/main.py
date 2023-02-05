import numpy as np
from utils import as_array

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None: # np.ndarray 형태로 강제하기위해
            if not isinstance(data, np.ndarray):
                raise TypeError('np.ndarray 자료형을 입력으로 넣어주세요')
        self.data = data
        self.grad = None # 미분값
        self.creator = None # 변수를 기준으로 함수는 creator 
    
    def set_creator(self, func) -> None:
        '''
        func은 함수
        '''
        self.creator = func # 함수
    
    def backward(self) -> None:
        funcs = [self.creator] # 함수를 리스트로
        '''
        역전파를 할때 첫 값은 1.0이라 grad값이 없을때 1.0을 넣어주려 했으나 다른 방법이 있었음
        여러 값이 들어 올땐 적합하지 않음
        '''
        if self.grad == None:
            self.grad = np.ones_like(self.data)

        while funcs:
            f = funcs.pop() # 함수를 하나하나 꺼내옴
            gys = [y.grad for y in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                
                if x.creator is not None:
                    funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError('순전파')
    
    def backward(self, gys):
        raise NotImplementedError('역전파')


