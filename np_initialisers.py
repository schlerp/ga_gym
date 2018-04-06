import numpy as np


# plain initialisation
def random_normal(shape, scale=1., mean=0.):
    return np.random.normal(mean, scale, shape)

def random_uniform(shape, scale=1.):
    return np.random.uniform(-scale, scale, shape)

def init_zeros(shape):
    return np.zeros(shape)

def init_ones(shape):
    return np.ones(shape)

def init_constant(shape, constant):
    return np.full(shape, constant)


# lecun initialisations
def lecun_normal(shape):
    sd = np.sqrt(3./shape[0])
    return np.random.normal(0., sd, shape)

def lecun_uniform(shape):
    limit = np.sqrt(3./shape[0])
    return np.random.uniform(-limit, limit, shape)


# he initialisations
def he_normal(shape):
    sd = np.sqrt(2./shape[0])
    return np.random.normal(0., sd, shape)

def he_uniform(shape):
    limit = np.sqrt(2./shape[0])
    return np.random.uniform(-limit, limit, shape)


# glorot initialisations
def glorot_normal(shape):
    sd = np.sqrt(2./np.sum(shape))
    return np.random.normal(0., sd, shape)

def glorot_uniform(shape):
    limit = np.sqrt(2./np.sum(shape))
    return np.random.uniform(-limit, limit, shape)


# genetic mutation
def ga_uniform(shape, limit):
    return np.random.uniform(-limit, limit, shape)

def ga_normal(shape, sd):
    return np.random.normal(0., sd, shape)