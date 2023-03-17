import numpy as np
import copy
from weights import *
from activations import *
from optimizers import *
from loss_file import *
# Base class for all neural networks

class NeuralNetwork():
  def __init__(self):
    pass
  def getParameters(self):
    pass
  def feedforward():
    pass
  def backpropogation():
    pass
  def test(self):
    pass
  def train(self):
    pass

# Neural Network for this particular neural network

class NN(NeuralNetwork):

  '''
  It contains the neural network desired for this application.
  Hidden layers, its sizes etc are all dynamic and can be set.
  
  '''
  def __init__(self, num_samples = 60000, input_size = 784, output_size = 10, num_hidden_layer = 3, hidden_layer_size=np.array([64,64,64]), data_name = "Fashion_mnsit", hidden_layer_activation="relu", output_layer_activation="softmax", weight_name="xavier",loss_name="cross_entropy"):
    self.num_samples = num_samples
    self.input_size = input_size
    self.output_size = output_size
    self.num_hidden_layer = num_hidden_layer
    self.hidden_layer_size = hidden_layer_size
    self.W, self.b = self.initializeWeights(weight_name)
    self.hidden_layer_activation = hidden_layer_activation
    self.output_layer_activation = output_layer_activation
    self.loss_name = loss_name
    self.activation_function = Activations()
    self.activate_hidden = self.activation_function.activate(hidden_layer_activation)
    self.activate_hidden_derivative = self.activation_function.derivate(hidden_layer_activation)
    self.activate_output = self.activation_function.activate(output_layer_activation)
    self.lossFunction = Loss(loss_name)
    self.loss = self.lossFunction.findLoss()
    self.lossDerivative = self.lossFunction.findDerivative()
    self.parameters = {
        "data_name":data_name,
        "num_samples":num_samples,
        "input_size":input_size,
        "output_size":output_size,
        "num_hidden_layer":num_hidden_layer,
        "hidden_layer_size":hidden_layer_size,
        "hidden_layer_activation":hidden_layer_activation,
        "output_layer_activation":output_layer_activation,
        "weight_init":weight_name,
        "loss_name":loss_name
    }

  def getParameters(self):
    return self.parameters

  def initializeWeights(self, weight_name):
    W = []
    b= []
    input_size = self.input_size
    weight_init = WeightInit(weight_name).initializeWeight()
    for i in range(self.num_hidden_layer):
      output_size = self.hidden_layer_size[i]
      W.append(weight_init((input_size, output_size ),0))
      b.append(weight_init((output_size, 1 ),1))
      input_size = output_size
    
    output_size = self.output_size

    W.append(weight_init((input_size, output_size),0))
    b.append(weight_init((output_size, 1),1))

    return W, b

  def calculateAccuracy(self, X, Y):
    Y_hat = self.feedforward(X)
    size = Y_hat.shape[1]
    score=0
    for i in range(size):
      if(np.argmax(Y_hat[:,i]) ==  Y[i]):
          score+=1

    return score/size * 100

  def feedforward(self, X):
    a = self.W[0].T @ X + self.b[0]
    hidden_layer_input = self.activate_hidden(a)

    for i in range(1,self.num_hidden_layer):
      a=self.W[i].T @ hidden_layer_input + self.b[i]
      hidden_layer_output=self.activate_hidden(a)
      hidden_layer_input = hidden_layer_output

    a=self.W[self.num_hidden_layer].T @ hidden_layer_input + self.b[self.num_hidden_layer]
    output = self.activate_output(a)

    return output


  def forwardpropogation(self, X):
    a_values=[]
    h_values=[]

    a = self.W[0].T @ X + self.b[0]
    hidden_layer_input = self.activate_hidden(a)
    
    a_values.append(a)
    h_values.append(hidden_layer_input)

    for i in range(1,self.num_hidden_layer):
      a=self.W[i].T @ hidden_layer_input + self.b[i]
      hidden_layer_output=self.activate_hidden(a)
      hidden_layer_input = hidden_layer_output
      a_values.append(a)
      h_values.append(hidden_layer_input)

    a=self.W[self.num_hidden_layer].T @ hidden_layer_input + self.b[self.num_hidden_layer]
    output = self.activate_output(a)
    a_values.append(a)
    h_values.append(output)

    return a_values,h_values

  def backpropogation(self, X, Y, a_values, h_values):
    size = len(h_values)
    data_size = Y.shape[0]
    delta_ak = self.lossDerivative(h_values[size - 1],Y)
    delta_W=[]
    delta_b=[]

    for k in range(size - 1,0,-1):
      delta_wk = h_values[k-1] @ delta_ak.T
      delta_bk = np.sum(delta_ak,axis=1)
      delta_W.append(delta_wk/data_size)
      delta_b.append(delta_bk.reshape(delta_bk.shape[0],1)/data_size)

      delta_hk = self.W[k] @ delta_ak
      # print(delta_hk.shape)
      # print(self.activation_function.sigmoidDerivative(a_values[k-1]).shape)
      delta_ak = np.multiply(self.activate_hidden_derivative(a_values[k-1]),delta_hk)

    delta_wk = X @ delta_ak.T
    delta_bk = np.sum(delta_ak,axis=1)
    delta_W.append(delta_wk/data_size)
    delta_b.append(delta_bk.reshape(delta_bk.shape[0],1)/data_size)

    return delta_W,delta_b
  
  def training(self, X, Y, X_val, Y_val ,epochs = 10 , weight_decay = 0 ,optimizer_name="gd", lr=0.01, batch_size=32,parameters=[]):
    optimize = Optimizer(optimizer_name).optimize()
    
    num_data = X.shape[1]
    np.random.seed(1)

    # Random shuffling of data
    indexes_for_batch = np.arange(num_data)
    np.random.shuffle(indexes_for_batch)

    return optimize(self, X, Y, X_val, Y_val , lr, epochs, batch_size, indexes_for_batch, weight_decay=weight_decay)
#     print(f'train accuracy: {self.calculateAccuracy(X,Y)}')