import numpy as np

# class having all the weight initializations
class WeightInit():
  '''
  It contains all weight initializations. initialize the object with name of initialization
  call object.initializeWeight() function to get the corresponding function, then pass shapes to get corresponding initializations
    
  The initializations are :
    -> zero
    -> random
    -> xavier
  
  '''
    
  def __init__(self,weight_name):
    self.weight_name = weight_name
    self.weight_dict={
        "zero":self.zeroInit,
        "random":self.randomInit,
        "xavier":self.xavierInit
    }

  def initializeWeight(self):
    return self.weight_dict[self.weight_name]
    
  
  def zeroInit(self, shape, flag=0):
    x = np.zeros(shape)
    return x

  def randomInit(self, shape, flag=0):
    x = np.random.normal(loc=0,scale=1,size=shape)
    return x

  
  def xavierInit(self, shape, flag=0):
    x = np.random.randn(*shape) * np.sqrt(2/shape[0]) if flag == 0 else self.zeroInit(shape)
    return x