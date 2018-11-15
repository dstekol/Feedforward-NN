import numpy as np


class NNmodel:
    def __init__(self, layers, r, act, actderiv):
        self.activ = np.vectorize(act)
        self.activ_deriv = np.vectorize(actderiv)
        self.weights = []
        self.layers = layers
        prevSize = 0
        for num in layers:
            self.weights.append(np.append(np.random.randn(num, prevSize), np.zeros((num, 1)), axis=1 ))
            prevSize = num
        self.weights[0] = []
        self.rate = r

    def forward_prop(self, inp):
        activations = []
        winputs = []
        act = np.array(inp)
        winp = np.array(inp)
        for i in range(1, len(self.weights)):
            w = self.weights[i]
            act = np.append(act, [[1]]*len(act), axis=1)
            activations.append(act)
            winputs.append(winp)
            winp = np.transpose(w.dot(np.transpose(act)))
            act = self.activ(winp)
        exp_scores = np.exp(winp)
        act =  exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        activations.append(act)
        winputs.append(winp)
        probs = act
        return probs, winputs, activations

    def backprop_iter(self, inp, outp): 
        probs, winputs, activations = self.forward_prop(inp)
        deltas = [np.zeros(layer) for layer in self.layers]
        prevSize = 0
        derivs = np.copy(self.weights)
        deltas[-1] = probs - outp
        derivs[-1] = activations[-2].T.dot(deltas[-1]).T
        for i in range(len(self.layers)-2, 0, -1):
            deltas[i] = deltas[i+1].dot(self.weights[i+1])[:,:-1] * self.activ_deriv(winputs[i])
            derivs[i] = activations[i-1].T.dot(deltas[i]).T
        for i in range(1, len(self.layers)):
            self.weights[i] -= self.rate * derivs[i]
        return derivs

    def backprop(self, trainData, mbSize, epochs):
        for j in range(epochs):
            print("epoch " + str(j))
            for i in range(0, len(trainData["input"]), mbSize):
                for x, y in zip(trainData["input"][i:i+mbSize], trainData["output"][i:i+mbSize]):
                    self.backprop_iter([x],y)
        print("done")

    def predict(self, inp):
        probs, _, _ = self.forward_prop(inp)
        return probs.argmax(axis=1)

    @staticmethod
    def convertToOutput(y, possible):
        outputs = []
        for i in y:
            outputs.append([(1 if i==x else 0) for x in possible])
        return outputs









