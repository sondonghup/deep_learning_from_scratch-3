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
            print('x 데이터', x)
            print('y 데이터', y)
            if isinstance(x, list) and isinstance(y, list):
                if y[0].grad == None:
                    # y.grad = np.array(1.0)
                    for x_value, y_value in zip (x, y):
                        print('셀프 데이터', self.data)
                        y_value.grad = np.ones_like(self.data) # data와 같은 크기의 1.0을 만들어준다.
                        print('aaaaaaaa',y_value.grad)
                x_value.grad = f.backward(y.grad)

                if x.creator is not None:
                    funcs.append(x.creator)
            else :
                if y == None:
                    # y.grad = np.array(1.0)
                    y.grad = np.ones_like(self.data) # data와 같은 크기의 1.0을 만들어준다.
                x.grad = f.backward(y.grad)

                if x.creator is not None:
                    funcs.append

class Function:
    def __call__(self, input):
        x = input.data
        print('x', x)
        if x.size > 1:
            y = []
            output_list = []
            self.input = []
            self.output = []
            for x_value in x:
                y = self.forward(x_value)
                print('----', y)
                output = Variable(as_array(y))
                output.set_creator(self)
                output_list.append(output)
                self.input.append(input)
            self.output = output_list

        else :
            y = self.forward(x)
            output = Variable(as_array(y)) # as_array는 계산이후 float64나 float32로 바뀌는 것때문에 추가함 왜냐하면 Variable에서 np.ndarray를 강제하는데 float값으로 나오면 이후 진행이 되지 않으므로
            output.set_creator(self)
            self.input = input
            self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError('순전파')
    
    def backward(self, gy):
        raise NotImplementedError('역전파')



