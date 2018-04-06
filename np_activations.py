import numpy as np

'''
Activations
-----------

this module houses all the activation functio classes. the classes themselves
can be used for calculate the activation function or the gradient of the 
activation function.

'''


class Activation(object):
    
    def __init__(self):    
        pass
    
    def forward(self, x):
        pass

    def __call__(self, x):
        pass


class Linear(Activation):
    
    def __init__(self, m=1.0, c=0.):
        self.m = m
        self.c = c
    
    def __call__(self, x):
        return (self.m * x) + self.c


class ReLU(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0)
    

class LeakyReLU(Activation):
    
    def __init__(self, stable=True, alpha=0.5):
        self.stable = stable
        self.alpha = alpha
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0) + x * self.alpha * (x <= 0)
    

class SchlerpReLU(Activation):
    def __init__(self, stable=True, m=1.0, c=0.0,
                 alpha_pos=0.1, alpha_neg=0.01, 
                 xstep_pos=1.0, xstep_neg=-1.0):
        self.stable = stable
        self.m = m
        self.c = c
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.xstep_pos = xstep_pos
        self.xstep_neg = xstep_neg
        self.ystep_pos = (self.m*self.xstep_pos)+self.c - (self.xstep_pos * self.alpha_pos)
        self.ystep_neg = (self.m*self.xstep_neg)+self.c - (self.xstep_neg * self.alpha_neg)
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return ((x * self.alpha_pos) + self.ystep_pos) * (x > self.xstep_pos) + \
               ((self.m*x)+self.c) * (np.logical_and(self.xstep_neg <= x, c <= self.xstep_pos)) + \
               ((x * self.alpha_neg) + self.ystep_neg) * (x < self.xstep_neg)


class SchlerpTanh(Activation):
    def __init__(self, stable=True, pos_alpha=0.1, neg_alpha=0.01):
        self.stable = stable
        self.pos_alpha = pos_alpha
        self.neg_alpha = neg_alpha
        self.pos_x_step = np.arctanh(np.sqrt(1-self.pos_alpha))
        self.neg_x_step = -np.arctanh(np.sqrt(1-self.neg_alpha))
        self.pos_y_step = np.tanh(self.pos_x_step) - \
                          (self.pos_x_step * self.pos_alpha)
        self.neg_y_step = np.tanh(self.neg_x_step) - \
                          (self.neg_x_step * self.neg_alpha)
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return ((x * self.pos_alpha) + self.pos_y_step) * (x > self.pos_x_step) + \
               np.tanh(x) * (np.logical_and(self.neg_x_step <= x, x <= self.pos_x_step)) + \
               ((x * self.neg_alpha) + self.neg_y_step) * (x < self.neg_x_step)
    

class ELU(Activation):
    
    def __init__(self, stable=True, alpha=1.0):
        self.stable = stable
        self.alpha = alpha
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return x * (x > 0) + self.alpha * (np.exp(x)-1) * (x <= 0)

class Sigmoid(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return 1 / (1 + np.exp(-x))
    
    
class Tanh(Activation):
    
    def __init__(self, stable=True):
        self.stable = stable
    
    def __call__(self, x):
        if self.stable:
            x = np.clip(x, -700, 700)
        return np.tanh(x)
    

class Softmax(Activation):    
    def __call__(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
