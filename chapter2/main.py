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
        self.generation = 0

    def print(self):
        print('data', self.data)
        print('grad', self.grad)
        print('creator', self.creator)
        print('generation', self.generation)
    
    def set_creator(self, func) -> None:
        '''
        func은 함수
        '''
        self.creator = func # 함수
        self.generation = func.generation + 1
    
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
                if x.grad == None:
                    x.grad = gx
                else :
                    x.grad += gx
                print(x.creator)
                print('xxx', x.generation)
                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        '''
        기존에 같은 변수를 여러번 사용하여 연산할경우 grad가 덮어 씌여지는 것을 해결 하기 위해 grad를 매번 더해주는 것을
        backward에서 진행 하였으나 완전 다른 연산을 할때에도 계속 grad가 더해져 다음값에 문제가 생김
        역전파를 할때마다 grad를 초기화 해주자
        '''
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.print()
            print(self.forward())
            output.set_creator(self)
           
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError('순전파')
    
    def backward(self, gys):
        raise NotImplementedError('역전파')


