import numpy as np
import copy
from weights import *

# classes having all the optimizers
class Optimizer():
    
  '''
  It contains all optimizers. initialize the object with name of optimizer
  call object.optimize() function to get the corresponding function, then pass appropriate parameter
  like training data, validation data, learning rate etc.
    
  The initializations are :
    -> gradient descent
    -> stochastic gradient descent
    -> momentum
    -> nesterov
    -> rmsprop
    -> adam
    -> nadam
  
  '''

  def __init__(self,optimizer_name="gd"):
    self.optimizer_name = optimizer_name
    self.optimizer_dict={
        "gd":self.gradient_descent,
        "sgd":self.gradient_descent,
        "momentum":self.momentum,
        "nag":self.nesterov,
        "rmsprop":self.rmsprop,
        "adam" : self.adam,
        "nadam" : self.nadam
    }

  def optimize(self):
    return self.optimizer_dict[self.optimizer_name]
    
  
  def gradient_descent(self, nn, X, Y, X_val, Y_val, lr, epochs, batch_size,indexes_for_batch,parameters = [], weight_decay = 0):

    num_data = X.shape[1]

    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []
    

    for epoch in range(epochs):
      for batch in range(0,num_data,batch_size):
        X_batch = X[:,indexes_for_batch[batch:batch + batch_size]]
        Y_batch = Y[indexes_for_batch[batch:batch + batch_size]]

        # self.W,self.b = self.initializeWeights()
        a_values,h_values = nn.forwardpropogation(X_batch)
        delta_W, delta_b = nn.backpropogation(X_batch,Y_batch,a_values, h_values)
        # print(np.sum(delta_W[0], axis = 0))
        for j in range(nn.num_hidden_layer + 1):
          nn.W[j] = nn.W[j] - lr * delta_W[nn.num_hidden_layer - j] - lr*weight_decay*nn.W[j]
          nn.b[j] = nn.b[j] - lr * delta_b[nn.num_hidden_layer - j]

      Y_hat = nn.feedforward(X)
      loss_value = nn.loss(Y_hat,Y,weight_decay,nn.W)
      print(f"epoch: {epoch} => loss = {loss_value}")
      y_val_predict = nn.feedforward(X_val)
      y_train_predict = nn.feedforward(X)

      validation_loss = nn.loss(y_val_predict,Y_val,weight_decay,nn.W)
      training_loss = nn.loss(y_train_predict,Y,weight_decay,nn.W)

      validation_accuracy = nn.calculateAccuracy(X_val, Y_val)
      training_accuracy = nn.calculateAccuracy(X, Y)

      
      val_loss_list.append(validation_loss)
      val_accuracy_list.append(validation_accuracy)
      train_loss_list.append(training_loss)
      train_accuracy_list.append(training_accuracy)
    
    return val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list
    
  def momentum(self, nn, X, Y, X_val, Y_val, lr, epochs, batch_size,indexes_for_batch,parameters = [0.9], weight_decay = 0):
    num_data = X.shape[1]
    ut_w,ut_b = nn.initializeWeights("zero")
    beta = parameters[0]

    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []

    for epoch in range(epochs):
      for batch in range(0,num_data,batch_size):
        X_batch = X[:,indexes_for_batch[batch:batch + batch_size]]
        Y_batch = Y[indexes_for_batch[batch:batch + batch_size]]

        # self.W,self.b = self.initializeWeights()
        a_values,h_values = nn.forwardpropogation(X_batch)
        delta_W, delta_b = nn.backpropogation(X_batch,Y_batch,a_values, h_values)
        # print(np.sum(delta_W[0], axis = 0))
        for j in range(nn.num_hidden_layer + 1):
          ut_w[j] = beta*ut_w[j] + delta_W[nn.num_hidden_layer - j]
          ut_b[j] = beta*ut_b[j] + delta_b[nn.num_hidden_layer - j] 

          nn.W[j] = nn.W[j] - lr * ut_w[j] - lr*weight_decay*nn.W[j]
          nn.b[j] = nn.b[j] - lr * ut_b[j]

      Y_hat = nn.feedforward(X)
      loss_value = nn.loss(Y_hat,Y,weight_decay,nn.W)
      print(f"epoch: {epoch} => loss = {loss_value}")
      y_val_predict = nn.feedforward(X_val)
      y_train_predict = nn.feedforward(X)

      validation_loss = nn.loss(y_val_predict,Y_val,weight_decay,nn.W)
      training_loss = nn.loss(y_train_predict,Y,weight_decay,nn.W)

      validation_accuracy = nn.calculateAccuracy(X_val, Y_val)
      training_accuracy = nn.calculateAccuracy(X, Y)

      
      val_loss_list.append(validation_loss)
      val_accuracy_list.append(validation_accuracy)
      train_loss_list.append(training_loss)
      train_accuracy_list.append(training_accuracy)
    
    return val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list
    
  def nesterov(self, nn, X, Y, X_val, Y_val, lr, epochs, batch_size,indexes_for_batch,parameters = [0.9], weight_decay = 0):
    num_data = X.shape[1]
    ut_w,ut_b = nn.initializeWeights("zero")
    beta = parameters[0]

    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []

    for epoch in range(epochs):
      for batch in range(0,num_data,batch_size):
        X_batch = X[:,indexes_for_batch[batch:batch + batch_size]]
        Y_batch = Y[indexes_for_batch[batch:batch + batch_size]]
  
        a_values,h_values = nn.forwardpropogation(X_batch)

        old_W = copy.deepcopy(nn.W)
        old_b = copy.deepcopy(nn.b)
        
        for k in range(nn.num_hidden_layer + 1):
          nn.W[k] = nn.W[k] - beta *  ut_w[k]
          nn.b[k] = nn.b[k] - beta *  ut_b[k]

        
        delta_W, delta_b = nn.backpropogation(X_batch,Y_batch,a_values, h_values)

        for j in range(nn.num_hidden_layer + 1):
          ut_w[j] = beta*ut_w[j] + delta_W[nn.num_hidden_layer - j]
          ut_b[j] = beta*ut_b[j] + delta_b[nn.num_hidden_layer - j] 

          nn.W[j] = old_W[j] - lr * ut_w[j] - lr*weight_decay*nn.W[j]
          nn.b[j] = old_b[j] - lr * ut_b[j]

      Y_hat = nn.feedforward(X)
      loss_value = nn.loss(Y_hat,Y,weight_decay,nn.W)
      print(f"epoch: {epoch} => loss = {loss_value}")
      y_val_predict = nn.feedforward(X_val)
      y_train_predict = nn.feedforward(X)

      validation_loss = nn.loss(y_val_predict,Y_val,weight_decay,nn.W)
      training_loss = nn.loss(y_train_predict,Y,weight_decay,nn.W)

      validation_accuracy = nn.calculateAccuracy(X_val, Y_val)
      training_accuracy = nn.calculateAccuracy(X, Y)

      
      val_loss_list.append(validation_loss)
      val_accuracy_list.append(validation_accuracy)
      train_loss_list.append(training_loss)
      train_accuracy_list.append(training_accuracy)
    
    return val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list
    


  def rmsprop(self, nn, X, Y,X_val, Y_val, lr, epochs, batch_size,indexes_for_batch,parameters = [0.9,0.1], weight_decay = 0):
    num_data = X.shape[1]
    vt_w,vt_b = nn.initializeWeights("zero")
    beta = parameters[0]
    epsilon = parameters[1]

    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []

    for epoch in range(epochs):
      for batch in range(0,num_data,batch_size):
        X_batch = X[:,indexes_for_batch[batch:batch + batch_size]]
        Y_batch = Y[indexes_for_batch[batch:batch + batch_size]]

        # self.W,self.b = self.initializeWeights()
        a_values,h_values = nn.forwardpropogation(X_batch)
        delta_W, delta_b = nn.backpropogation(X_batch,Y_batch,a_values, h_values)
        # print(np.sum(delta_W[0], axis = 0))
        for j in range(nn.num_hidden_layer + 1):
          vt_w[j] = beta*vt_w[j] + (1 - beta) * np.multiply(delta_W[nn.num_hidden_layer - j],delta_W[nn.num_hidden_layer - j]) 
          vt_b[j] = beta*vt_b[j] + (1 - beta) * np.multiply(delta_b[nn.num_hidden_layer - j],delta_b[nn.num_hidden_layer - j])

          nn.W[j] = nn.W[j] - np.divide(lr * delta_W[nn.num_hidden_layer - j],np.sqrt(vt_w[j] + epsilon)) - lr*weight_decay*nn.W[j]
          nn.b[j] = nn.b[j] - np.divide(lr * delta_b[nn.num_hidden_layer - j],np.sqrt(vt_b[j] + epsilon))

      Y_hat = nn.feedforward(X)
      loss_value = nn.loss(Y_hat,Y,weight_decay,nn.W)
      print(f"epoch: {epoch} => loss = {loss_value}")
      y_val_predict = nn.feedforward(X_val)
      y_train_predict = nn.feedforward(X)

      validation_loss = nn.loss(y_val_predict,Y_val,weight_decay,nn.W)
      training_loss = nn.loss(y_train_predict,Y,weight_decay,nn.W)

      validation_accuracy = nn.calculateAccuracy(X_val, Y_val)
      training_accuracy = nn.calculateAccuracy(X, Y)

      
      val_loss_list.append(validation_loss)
      val_accuracy_list.append(validation_accuracy)
      train_loss_list.append(training_loss)
      train_accuracy_list.append(training_accuracy)
    
    return val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list
    
  def adam(self, nn, X, Y, X_val, Y_val, lr, epochs, batch_size,indexes_for_batch,parameters = [0.9,0.99,0.1], weight_decay = 0):
    num_data = X.shape[1]
    vt_w,vt_b = nn.initializeWeights("zero")
    mt_w,mt_b = nn.initializeWeights("zero")
    beta1 = parameters[0]
    beta2 = parameters[1]
    epsilon = parameters[2]

    t=0

    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []

    for epoch in range(epochs):
      for batch in range(0,num_data,batch_size):
        t+=1
        X_batch = X[:,indexes_for_batch[batch:batch + batch_size]]
        Y_batch = Y[indexes_for_batch[batch:batch + batch_size]]

        # self.W,self.b = self.initializeWeights()
        a_values,h_values = nn.forwardpropogation(X_batch)
        delta_W, delta_b = nn.backpropogation(X_batch,Y_batch,a_values, h_values)
        # print(np.sum(delta_W[0], axis = 0))
        for j in range(nn.num_hidden_layer + 1):
          mt_w[j] = beta1 * mt_w[j] + (1 - beta1) * delta_W[nn.num_hidden_layer - j]
          mt_b[j] = beta1 * mt_b[j] + (1 - beta1) * delta_b[nn.num_hidden_layer - j]

          mt_w_dash = mt_w[j] / (1 - beta1 ** t)
          mt_b_dash = mt_b[j] / (1 - beta1 ** t)


          vt_w[j] = beta2*vt_w[j] + (1 - beta2) * np.multiply(delta_W[nn.num_hidden_layer - j],delta_W[nn.num_hidden_layer - j]) 
          vt_b[j] = beta2*vt_b[j] + (1 - beta2) * np.multiply(delta_b[nn.num_hidden_layer - j],delta_b[nn.num_hidden_layer - j])

          vt_w_dash = vt_w[j] / (1 - beta2 ** t)
          vt_b_dash = vt_b[j] / (1 - beta2 ** t)           

          nn.W[j] = nn.W[j] - np.divide(lr * mt_w_dash,np.sqrt(vt_w_dash + epsilon)) - lr*weight_decay*nn.W[j]
          nn.b[j] = nn.b[j] - np.divide(lr * mt_b_dash,np.sqrt(vt_b_dash + epsilon))

      Y_hat = nn.feedforward(X)
      loss_value = nn.loss(Y_hat,Y,weight_decay,nn.W)
      print(f"epoch: {epoch} => loss = {loss_value}")
      y_val_predict = nn.feedforward(X_val)
      y_train_predict = nn.feedforward(X)

      validation_loss = nn.loss(y_val_predict,Y_val,weight_decay,nn.W)
      training_loss = nn.loss(y_train_predict,Y,weight_decay,nn.W)

      validation_accuracy = nn.calculateAccuracy(X_val, Y_val)
      training_accuracy = nn.calculateAccuracy(X, Y)

      
      val_loss_list.append(validation_loss)
      val_accuracy_list.append(validation_accuracy)
      train_loss_list.append(training_loss)
      train_accuracy_list.append(training_accuracy)
    
    return val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list
    
  def nadam(self, nn, X, Y,X_val, Y_val, lr, epochs, batch_size,indexes_for_batch,parameters = [0.9, 0.999, 0.1], weight_decay = 0):
    num_data = X.shape[1]
    vt_w,vt_b = nn.initializeWeights("zero")
    mt_w,mt_b = nn.initializeWeights("zero")
    beta1 = parameters[0]
    beta2 = parameters[1]
    epsilon = parameters[2]

    t=0

    val_loss_list = []
    train_loss_list = []
    val_accuracy_list = []
    train_accuracy_list = []

    for epoch in range(epochs):
      for batch in range(0,num_data,batch_size):
        t+=1
        X_batch = X[:,indexes_for_batch[batch:batch + batch_size]]
        Y_batch = Y[indexes_for_batch[batch:batch + batch_size]]

        # self.W,self.b = self.initializeWeights()
        a_values,h_values = nn.forwardpropogation(X_batch)
        delta_W, delta_b = nn.backpropogation(X_batch,Y_batch,a_values, h_values)
        # print(np.sum(delta_W[0], axis = 0))
        for j in range(nn.num_hidden_layer + 1):
          mt_w[j] = beta1 * mt_w[j] + (1 - beta1) * delta_W[nn.num_hidden_layer - j]
          mt_b[j] = beta1 * mt_b[j] + (1 - beta1) * delta_b[nn.num_hidden_layer - j]

          mt_w_dash = mt_w[j] / (1 - beta1 ** t)
          mt_b_dash = mt_b[j] / (1 - beta1 ** t)


          vt_w[j] = beta2*vt_w[j] + (1 - beta2) * np.multiply(delta_W[nn.num_hidden_layer - j],delta_W[nn.num_hidden_layer - j]) 
          vt_b[j] = beta2*vt_b[j] + (1 - beta2) * np.multiply(delta_b[nn.num_hidden_layer - j],delta_b[nn.num_hidden_layer - j])

          vt_w_dash = vt_w[j] / (1 - beta2 ** t)
          vt_b_dash = vt_b[j] / (1 - beta2 ** t)           

          w_update_numerator = lr * (beta1 * mt_w_dash + ((1 - beta1)* delta_W[nn.num_hidden_layer - j]/(1 - beta1 ** t)))
          b_update_numerator = lr * (beta1 * mt_b_dash + ((1 - beta1)* delta_b[nn.num_hidden_layer - j]/(1 - beta1 ** t)))

          nn.W[j] = nn.W[j] - np.divide(w_update_numerator,np.sqrt(vt_w_dash + epsilon)) - lr*weight_decay*nn.W[j]
          nn.b[j] = nn.b[j] - np.divide(b_update_numerator,np.sqrt(vt_b_dash + epsilon))

      Y_hat = nn.feedforward(X)
      loss_value = nn.loss(Y_hat,Y,weight_decay,nn.W)
      print(f"epoch: {epoch} => loss = {loss_value}")
      y_val_predict = nn.feedforward(X_val)
      y_train_predict = nn.feedforward(X)

      validation_loss = nn.loss(y_val_predict,Y_val,weight_decay,nn.W)
      training_loss = nn.loss(y_train_predict,Y,weight_decay,nn.W)

      validation_accuracy = nn.calculateAccuracy(X_val, Y_val)
      training_accuracy = nn.calculateAccuracy(X, Y)

      
      val_loss_list.append(validation_loss)
      val_accuracy_list.append(validation_accuracy)
      train_loss_list.append(training_loss)
      train_accuracy_list.append(training_accuracy)
    
    return val_loss_list,val_accuracy_list,train_loss_list,train_accuracy_list
    