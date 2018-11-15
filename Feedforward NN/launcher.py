import neural
import numpy as np 
import sklearn 
import sklearn.datasets 
import math

 
np.random.seed(3) 
X, y = sklearn.datasets.make_moons(200, noise=0.20)

def sigmoid(v):
    return 1/(1+math.exp(-v))

def sigmoid_deriv(v):
    s = sigmoid(v)
    return s*(1-s)

def tanh(v):
    return np.tanh(v)

def tanh_deriv(v):
    return 1-(tanh(v)**2)

def relu(v):
    return max(0, v)

def relu_deriv(v):
    return 0 if v<0 else 1


m = neural.NNmodel([2, 5, 2], .1, lambda x: tanh(x), lambda x: tanh_deriv(x))
traindata = {"input": X, "output":  neural.NNmodel.convertToOutput(y, [0,1])}
m.backprop(traindata, epochs=20, mbSize=10)

numRight = 0
for pred, actual in zip(m.predict(X), y):
    if(pred==actual):
        numRight+=1
trainingAccuracy = numRight / len(X)
print("Training Accuracy: " + str(trainingAccuracy))



