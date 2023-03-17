import numpy as np

# class having all losses and its derivatives
class Loss():
  '''
  It contains all Loss and its corresponding derivative with last layer (a_L), assuming last layer activation is softmax.
  Initialize the object with name of loss.
  call object.findLoss() function to get the corresponding function, then pass appropriate parameter
  like Y, Y_hat etc.
    
  The initializations are :
    -> cross_entropy
    -> mean_squared_error
  
  '''
    
    
  def __init__(self,loss_name):
    self.loss_dict = {
        'cross_entropy':self.crossEntropy,
        'mean_squared_error':self.mse
    }
    self.loss_derivative_dict = {
        'cross_entropy':self.crossEntropyDerivativeWithL,
        'mean_squared_error':self.mseDerivativeWithL
    }
    self.loss_name = loss_name

  def findLoss(self):
    return self.loss_dict[self.loss_name]
  
  def findDerivative(self):
    return self.loss_derivative_dict[self.loss_name]

  def findOneHotVector(self,Y_hat, Y):
    vector = np.zeros(Y_hat.shape)
    for i in range(Y_hat.shape[1]):
      vector[:,i][Y[i]] = 1
    
    return vector
    
  def crossEntropy(self ,Y_hat, Y, weight_decay=0,W=[]):
    loss=0
    num_samples = Y_hat.shape[1]
    for i in range(num_samples):
      loss+=np.log(Y_hat[:,i][Y[i]]  if Y_hat[:,i][Y[i]] != 0 else 1e-9)
    
    decay_loss = 0
    for i in range(len(W)):
        decay_loss+=np.sum(W[i] ** 2)


    return (-loss + (weight_decay * decay_loss)/2)/num_samples

  def crossEntropyDerivativeWithL(self,Y_hat, Y):
    return -(self.findOneHotVector(Y_hat,Y) - Y_hat)

  def mse(self, Y_hat, Y,weight_decay=0,W=[]):
    m,n = Y_hat.shape
    diff_matrix = (Y_hat - Y)**2
    loss = np.sum(diff_matrix)
    
    decay_loss = 0
    for i in range(len(W)):
        decay_loss+=np.sum(W[i] ** 2)
        
    
    return (loss/(m*n)) + ((weight_decay * decay_loss)/2)/n

  def mseDerivativeWithL(self, Y_hat , Y,weight_decay=0,W=[]):
    derivative = np.zeros(Y_hat.shape)
    num_rows,num_samples = Y_hat.shape
    one_hot_Y = self.findOneHotVector(Y_hat,Y)
    for m in range(num_samples):
        for j in range(num_rows):
            s=0
            for i in range(num_rows):
                if j == i:
                    s+=(Y_hat[i,m] - one_hot_Y[i,m])*Y_hat[i,m]*(1 - Y_hat[j,m])
                else:
                    s+=(Y_hat[i,m] - one_hot_Y[i,m])*Y_hat[i,m]*(-Y_hat[j,m])
                
            derivative[j,m] = 2*s
    return derivative
    