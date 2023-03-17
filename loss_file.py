import numpy as np

# class having all losses and its derivatives
class Loss():
  def __init__(self,loss_name):
    self.loss_dict = {
        'cross_entropy':self.crossEntropy,
        'mean_squared_error':self.mse
    }
    self.loss_name = loss_name

  def findLoss(self):
    return self.loss_dict[self.loss_name]
    
  def crossEntropy(self ,Y_hat, Y, weight_decay=0,W=[]):
    loss=0
    num_samples = Y_hat.shape[1]
    for i in range(num_samples):
      loss+=np.log(Y_hat[:,i][Y[i]]  if Y_hat[:,i][Y[i]] != 0 else 1e-9)
    
    decay_loss = 0
    for i in range(len(W)):
        decay_loss+=np.sum(W[i] ** 2)


    return (-loss + (weight_decay * decay_loss)/2)/num_samples

  def crossEntropyDerivative(self,Y_hat, Y):
    derivative = np.zeros(Y_hat.shape)
    for i in range(Y_hat.shape[1]):
      derivative[:,i][Y[i]] = 1/(Y_hat[:,i][Y[i]])
    return derivative

  def mse(self, Y_hat, Y):
    m,n = Y_hat.shape
    diff_matrix = (Y_hat - Y)**2
    loss = np.sum(diff_matrix)
    return loss/(m*n)

  def mseDerivative(self, Y_hat , Y):
    derivative = np.zeros(Y_hat.shape)
    for i in range(Y_hat.shape[1]):
      derivative[:,i][Y[i]] = 1/(Y_hat[:,i][Y[i]])
    return derivative
    