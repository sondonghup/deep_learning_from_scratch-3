from main import Function

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y, )

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)