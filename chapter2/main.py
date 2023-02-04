import numpy as np
from utils import as_array

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        print("data shape", data.size)
        if data is not None: # np.ndarray 형태로 강제하기위해
            # data = as_array(data)
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
        print('funcs', funcs)

        '''
        역전파를 할때 첫 값은 1.0이라 grad값이 없을때 1.0을 넣어주려 했으나 다른 방법이 있었음
        여러 값이 들어 올땐 적합하지 않음
        '''
        # if self.grad == None:
        #     self.grad = np.array(1.0) 

        while funcs:
            f = funcs.pop() # 함수를 하나하나 꺼내옴
            x, y = f.input, f.output
            if y == None:
                # y.grad = np.array(1.0)
                y.grad = np.ones_like(self.data) # data와 같은 크기의 1.0을 만들어준다.
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append

class Function:
    def __call__(self, inputs):
        print('tqlasdfe', inputs)
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError('순전파')
    
    def backward(self, gys):
        raise NotImplementedError('역전파')



