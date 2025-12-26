#import numpy as np
class Logisic_Regression:
  def __init__(self,x,y,output = 1):
    self.x = x
    self.y = y
    self.output = output
    self.w = None
    self.b = None

  def sigmoid(self,z):
    return 1/(1+ np.exp(-z))

  def parameters_init(self):
      w = np.random.randn(self.x.shape[1],self.output)
      b = np.zeros((1,self.output))
      return w,b

  def train(self,epoch = 1200,lr=0.01,show_loss=False):
    self.w,self.b = self.parameters_init()
    for i in range(epoch):
      z = np.matmul(self.x,self.w) + self.b
      a = self.sigmoid(z)
      loss = -np.mean(self.y*np.log(a)+(1 - self.y)*np.log(1-a))
      dz = a - self.y
      dw = np.matmul(self.x.T,dz)
      db = np.sum(dz,axis=0,keepdims=True)
      self.w = self.w - lr * dw
      self.b = self.b - lr * db
      if show_loss:
        if i%100==0:
          print(f"loss is :- {loss}")
    return self.w,self.b

  def predict(self,x):
    z = np.matmul(x,self.w) + self.b
    a = self.sigmoid(z)
    return a

  def BCELoss(self,y_pred,y):
    return -np.mean(y*np.log(y_pred) + (1 - y)*np.log(1 - y_pred))

x = np.random.randn(5,4)
y = np.array([[1],[0],[0],[1],[0]])

logReg = Logisic_Regression(x,y)
logReg.train()
x_test = np.random.randn(2,4)
y_test = np.array([[1],[0]])
y_pred = logReg.predict(x_test)
logReg.BCELoss(y_pred,y_test)
