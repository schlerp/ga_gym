import numpy as np
from scipy import signal as sg


from np_activations import *
from np_initialisers import *


# exceptions
class LayerNotCompiled(Exception):
    def __init__(self):
        self.message = 'Layer has not been compiled!\n'
        self.message += 'Please run the `compile` method of this layer'

class IncorrectShapeException(Exception):
    def __init__(self, expected_shape_rank, actual_shape_rank):
        self.message = 'Shape must be rank {}\n'.format(expected_shape_rank)
        self.message += 'Recieved rank {}'.format(actual_shape_rank)


# Layers
class Layer(object):
    def __init__(self):
        pass
    def compile(self):
        pass
    def forward(self):
        pass


class InputLayer(Layer):
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size
    def forward(self, x):
        return x


class Dense(Layer):
    def __init__(self, in_layer, output_size, act=Linear()):
        self.input_size = in_layer.output_size
        self.output_size = output_size
        self.weight_shape = (self.input_size, self.output_size)
        self.bias_shape = (self.output_size,)
        self.activation = act
        self.compiled = False
    
    def compile(self, weight_init=he_uniform, bias_init=init_zeros):
        self.weights = weight_init(self.weight_shape)
        self.biass = bias_init(self.bias_shape)
        self.compiled = True
    
    def forward(self, x):
        # check if layer has been initialised
        if not self.compiled:
            raise LayerNotCompiled()
        
        # run forward pass
        self.z = np.dot(x, self.weights) + self.biass
        self.y = self.activation(self.z)
        return self.y


class Conv2D(Layer):
    def __init__(self, in_layer, num_filters, fshape, stride=1, act=Linear()):
        self.expected_shape_rank = 3
        self.input_size = in_layer.output_size
        self.num_filters = num_filters
        self.filter_shape = fshape
        self.filters_shape = (*self.filter_shape, self.num_filters)
        self.stride = stride
        self.activation = act
        self.compiled = False
    
    def compile(self, filter_init=glorot_uniform, mode='same'):
        self.filters = filter_init(self.filters_shape)
        self.mode = mode
        self.compiled = True
    
    def forward(self, x):
        # check if layer has been initialised
        if not self.compiled:
            raise LayerNotCompiled()
        
        # check if shape is correct
        actual_shape_rank = len(x.shape)
        if actual_shape_rank != self.expected_shape_rank:
            raise IncorrectShapeException(self.expected_shape_rank,
                                          actual_shape_rank)
        
        # run convolution
        self.z = sg.convolve(x, self.filters, mode=self.mode)
        self.y = self.activation(self.z)
        
        return self.y


class Flatten(Layer):
    def forward(self, x):
        return x.flatten()

if __name__ == '__main__':
    import skimage
    import skimage.io
    
    img_file = r'C:\Users\Pat12\Pictures\sprocket.jpg'
    
    img = skimage.io.imread(img_file, as_grey=True)
    
    img = img.reshape((*img.shape, 1))
    
    in_layer = InputLayer(img.shape)
    
    conv = Conv2D(in_layer, 1, (3, 3), act=Sigmoid())
    conv.compile()
    
    conv.filters = np.array([[[-0.], [-1.], [-0.]],
                             [[-1.], [ 1.], [-1.]],
                             [[-0.], [-1.], [-0.]]])
    
    img_out = conv.forward(img)
    
    img_out = img_out.reshape(img_out.shape[0:2])
    
    skimage.io.imsave(r'C:\Users\Pat12\Pictures\sprocket_conv.jpg', img_out)