import numpy as np
from main import Variable
from add_func import add
from cubic_equation_func import cubic_equation
from square_func import square
from exp_func import exp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--func', required=True)
    args = parser.parse_args()
    print(args.func)

    if args.func == 'add':
        x0, x1 = map(int, input().split())
        x0 = Variable(np.array(x0))
        x1 = Variable(np.array(x1))
        y = add(x0, x0)
        y.backward()
        print(y.data)
        print(x0.grad)
        print(x1.grad)


    elif args.func == 'cubic':
        x = int(input())
        x = Variable(np.array(x))
        y = cubic_equation(x)
        y.backward()
        print(x.grad)
        print(y)

    elif args.func == 'square':
        x = int(input())
        x = Variable(np.array(x))
        y = square(x)
        y.backward()
        print(x.grad)
        print(y)

    elif args.func == 'exp':
        x = int(input())
        x = Variable(np.array(x))
        y = exp(x)
        y.backward()
        print(x.grad)
        print(y)
    
    elif args.func == 'custom':
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(2.0))
        y0 = square(x0)
        y1 = add(y0, y0)
        y1.backward()
        print(y1.grad)
