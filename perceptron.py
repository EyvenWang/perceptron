# import dependencies
import keras
from keras.datasets import mnist
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm

'''
Process the dataset
'''

# import dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess dataset. input image size is 28 * 28, make them flat, meaning they should be of size (1, 28*28)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Adding bias
train_bias = np.ones((60000, 1))
test_bias = np.ones((10000, 1))

x_train = np.append(train_bias, x_train, axis=1)
x_test = np.append(test_bias, x_test, axis=1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizae dataset. The range of pixel value is [0, 255], make them in [0, 1]
x_train /= 255
x_test /= 255

print(y_train.shape,)
# one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train, y_train, x_test, y_test = x_train[:20000], y_train[:20000], x_test[:2000], y_test[:2000]
print(x_train.shape, y_test.shape)

'''
Build a one-layer MLP
'''
class MLP():
  def __init__(self):
    self.lr = 0.001 # learning rate
    self.iters = 100 # epoches
    self.hidden_dim = 10 # number of neurons in hidden layer
    self.weight = np.random.randn(28*28+1, self.hidden_dim)
    self.softmax = nn.Softmax()
    
  def forward(self, x):
    x11 = np.dot(x, self.weight)
    x12 = self.softmax(torch.tensor(x11)).numpy()

    return x11, x12

  def train_CE(self, x, target):
     for i in tqdm(range(self.iters)):
       x11, x12 = self.forward(x)

       # calculate gradient and backprob
       d_loss_x11 = (x12 - target)
       d_loss_weights = np.dot(x.T, d_loss_x11)

       # update weights and bias
       self.weight -= self.lr * d_loss_weights 

  def train_perceptron(self, x, target): 
    self.weight = np.random.randn(28*28+1,)
    target = np.argmax(target, axis=1)


    for i in tqdm(range(self.iters)):
      for label in range(10):
        for data, y in zip(x, target):
          if i == y:
            y = 1
          else:
            y = -1

          pred = np.dot(data, self.weight) * y

          if pred <= 0:
            self.weight += data * y * self.lr


  def test_perceptron(self, x, y):
    total = dict()
    correct = dict()
    for i in range(10):
      total[i] = 0
      correct[i] = 0

    target = np.argmax(y, axis=1)
    for label in range(10):
      for data, y in zip(x, target):
        if label == y:
          pred = np.dot(data, self.weight) * 1
        else:
          pred = np.dot(data, self.weight) * (-1)

        if pred > 0:
          correct[label] = correct[label] + 1

        total[label] = total[label] + 1

    overall = 0
    for i in range(10):
      rate = correct[i]/total[i]
      overall += rate
      print("label {}'s accracy: {}".format(i, correct[i]/total[i]))

    print(" Overall Test Accuracy:", overall/10)

       
  
  def test_CE(self, x, y):
    output = self.forward(x)[-1]
    preds = np.argmax(output, axis=1)
    target = np.argmax(y, axis=1)

    total = dict()
    correct = dict()
    for i in range(10):
      total[i] = 0
      correct[i] = 0

    for (i, j) in zip(preds, target):
      if i == j:
        correct[j] = correct[j] + 1
      
      total[j] = total[j] + 1

    for i in range(10):
      print("label {}'s accracy: {}".format(i, correct[i]/total[i]))

    match = preds == target
    print(" Overall Test Accuracy:", np.sum(match)/preds.shape[0])

mlp = MLP()
mlp.train_perceptron(x_train, y_train)
mlp.test_perceptron(x_test, y_test)

mlp = MLP()
mlp.train_CE(x_train, y_train)
mlp.test_CE(x_test, y_test)
