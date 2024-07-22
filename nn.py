import random
from crudgrad.engine import Value

class Neuron:

  def __init__(self, nin):
    """
    nin = number of inputs to the neuron
    """
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # one weight for every input into this neuron
    self.b = Value(random.uniform(-1, 1)) # one bias for the neuron itself

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # 2nd arg tells sum where to start. 0.0 by default. This is equivalent to adding the act function to self.b
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]

class Layer:

  def __init__(self, nin, nout):
    """
    nin = number of inputs to each neuron
    nout = number of neurons in this layer
    """
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
    # above line is identical to the code below
    # params = []
    # for neuron in self.neurons:
    #   ps = neuron.parameters()
    #   params.extend(ps) # append adds a single element to a list. extend adds a list of elements to a list
    # return params

class MLP:

  def __init__(self, nin, nouts):
    """
    nin = number of inputs to the MLP
    nouts = list of number of neurons in each layer
    """
    sz = [nin] + nouts  # creates a list of [no. of inputs, no. of layer 1 neurons, no. of layer 2 neurons, ...]
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] # creates each layer of neurons sequentially

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]