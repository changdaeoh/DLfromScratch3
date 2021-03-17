import weakref
import numpy as np
from dezero import cuda
import dezero.functions as F
from dezero.core import Parameter



# =============================================================================
# Layer (base class)
# =============================================================================

class Layer:
    def __init__(self):
        self._params = set()                
        
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):  # add Layer
            self._params.add(name)
        super().__setattr__(name, value)
    
    # input을 forward 연산에 전달
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    # 구체적인 연산은 자식클래스에서 구현
    def forward(self, inputs):
        raise NotImplementedError()
    
    # 해당 레이어 인스턴스가 가진 파라미터들에 접근
    def params(self):
        for name in self._params:
            # name을 키(속성 이름)로써 저장된 value object(속성 값)에 접근
            obj = self.__dict__[name] 
            
            if isinstance(obj, Layer):  # yield from layer       
                yield from obj.params() # recurrent call
            else:
                yield obj
    
    # 모든 매개변수들의 그레디언트를 초기화            
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
            
    def _flatten_params(self, params_dict, parent_key = ""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
    
    # gpu 지원
    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================


class Linear(Layer):
    # input size를 미리 지정하지 않아도 입력된 데이터의 shape로부터 유추할 수 있다!
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name = "W")
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype = dtype), name = 'b')
        
    # weight initialization
    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

        
    def forward(self, x):
        # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            
            xp = cuda.get_array_module(x)
            self._init_W(xp)
            
        y = F.linear(x, self.W, self.b)
        return y