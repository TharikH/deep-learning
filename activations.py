import numpy as np
import copy

#class having all required activations

class Activations():
  '''
  It contains all activations required with its derivatives.
  call object.activate(name) function to get the corresponding function, then pass vectors/matrix/values on to these fucntions
  similarly call object.derivative(name) to get derivative function.
  The names are :
  -> sigmoid
  -> softmax
  -> tanh
  -> relu
  -> identity
  
  '''
  def __init__(self):
    self.activation_dict={
        "sigmoid":self.sigmoid,
        "softmax":self.softmax,
        "tanh":self.tanh,
        "relu":self.relu,
        "identity":self.identity
    }
    self.derivative_dict={
        "sigmoid":self.sigmoidDerivative,
        "softmax":self.softmaxDerivative,
        "tanh":self.tanhDerivative,
        "relu":self.reluDerivative,
        "identity":self.identity
    }

  def activate(self, activation_function = "sigmoid"):
    return self.activation_dict[activation_function]

  def derivate(self,activation_function = "sigmoid"):
    return self.derivative_dict[activation_function]

  def sigmoid(self, x):
    z = x.copy()
    z[x < 0] = np.exp(x[x < 0])/(1 + np.exp(x[x<0]))
    z[x >= 0] = 1/(1+np.exp(-x[x >= 0]))
    return z

  def softmax(self, x):
    max_element = np.max(x,axis=0)
    z = np.exp(x - max_element)
    total = sum(z)
    z = z/total
    return z
  
  def tanh(self, x):
    return np.tanh(x)

  def identity(self,x):
    return x

  def identityDerivative(self, x):
    return np.ones(x.shape)

  def tanhDerivative(self, x):
    z = self.tanh(x)
    return 1 - z**2

  def softmaxDerivative(self,x):
    pass
  
  def sigmoidDerivative(self,x):
    z = self.sigmoid(x)
    return  z*(1 - z)
  
  def relu(self,x):
    return np.maximum(x,0)
  
  def reluDerivative(self,x):
    z = x.copy()
    z[x < 0]=0
    z[x > 0]=1
    return z