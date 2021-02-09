import numpy as np
import dezero
from dezero.core import Function, as_variable
from dezero import cuda, utils


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x) 
        return gx

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y )
        return gx
    
def tanh(x):
    return Tanh()(x)

#---------------------------------------------------------------------------------------
# reshape, transpose
#---------------------------------------------------------------------------------------

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape # 변형 목표가 되는 shape
    
    def forward(self, x):
        self.x_shape = x.shape # x의 original shape를 기억
        y = x.reshape(self.shape) # 넘파이의 reshape함수 이용
        return y
    
    # 연산 없이 형상만 변형 (원래 x의 shape대로)
    def backward(self, gy):
        return reshape(gy, self.x_shape) # gy는 Variable 객체이므로 별도의 reshape함수 만들어야함

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)




class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = dezero.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
