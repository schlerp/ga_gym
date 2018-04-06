import numpy as np


from np_layers import InputLayer, Dense, Conv2D
from np_activations import Tanh, LeakyReLU, Softmax
from np_initialisers import ga_normal, ga_uniform


class Network(object):
    pass


class CompressedNetwork(Network):
    def __init__(self, seed=None, other_seeds=None):
        self.seed = seed if seed else np.random.randint(0, 2**8-1)
        self.other_seeds = other_seeds if other_seeds else []


class GeneticNetwork1D(Network):
    def __init__(self, input_size, output_size, seed, width=64, num_hidden=3):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.evolve_layers = []
        self.seed = seed
        self.other_seeds = []
        
        # input layer
        self.input_layer = InputLayer(self.input_size)
        self.layers.append(self.input_layer)
        
        # hidden layers
        for i in range(num_hidden):
            layer = Dense(self.layers[-1], width, LeakyReLU())
            self.layers.append(layer)
            self.evolve_layers.append(layer)
            
        # output layer
        self.output_layer = Dense(self.layers[-1], output_size, Softmax())
        self.layers.append(self.output_layer)
        self.evolve_layers.append(self.output_layer)
    
    def compile(self):
        np.random.seed(self.seed)
        for layer in self.layers:
            layer.compile()
    
    def forward(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x)
        return x.argmax()
    
    def evolve(self, new_seed, ga_init=ga_normal, ga_sigma=1.0):
        self.other_seeds.append(new_seed)
        np.random.seed(new_seed)
        for layer in self.evolve_layers:
            layer.weights += ga_init(layer.weights.shape, ga_sigma)

