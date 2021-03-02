import numpy as np
from dezero import Layer
import dezero.functions as F
import dezero.layers as L
from dezero import utils


# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)   # layer name과 layer object를 속성으로써 반복저장
            self.layers.append(layer)          # layers 리스트 속성에 layer object 반복저장
            
    def forward(self, x):
        for l in self.layers[:-1]:             # output layer에는 activation을 적용하지 않기 때문
            x = self.activation(l(x))
        return self.layers[-1](x)        
