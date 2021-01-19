import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        # use 'generation' variable to control the flow of backward gradients
        self.generation = 0 

    def set_creator(self, func):
        self.creator = func
        # variable & function both have 'generation'
        self.generation = func.generation + 1 

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # funcs & seen_set : space that functions will be stacked
        funcs = [] 
        seen_set = set() # set data type -> with no duplicates elements
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x: x.generation) # sorting by generation
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop() # taken out sequentially from the higher geneeration
            gys = [output.grad for output in f.outputs] # multiple gradients
            gxs = f.backward(*gys) 
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                # prevent overwriting of gradient when using the same variable instance
                if x.grad is None: 
                    x.grad = gx 
                else :
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)

    # prevent accumalting of gradient when using the same variable instance
    def cleargrad(self):
        self.grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): 
        xs = [x.data for x in inputs] # multiple inputs
        ys = self.forward(*xs) 
        if not isinstance(ys, tuple): # Tupleizating outputs
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] 
        
        self.generation = max([x.generation for x in inputs]) # setting 'generation' of function
        for output in outputs: 
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0] # to return 1 or many outputs correctly
    
    def forward(self, xs):
        raise NotImplementedError()
        
    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    # multiple intputs
    def forward(self, x0, x1): 
        y = x0 + x1
        return y
    
    # multiple backward gradients
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data # to deal with tupleized single input
        gx = 2 * x * gy
        return gx

def square(x):
    f = Square()
    return f(x)


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)