import numpy as np

from layers import *
from gradient import numerical_gradient
from collections import OrderedDict


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size):
        self.params = {}
        self.layers = OrderedDict()
        hidden_size = 0
        for index, hidden_size in enumerate(hidden_size_list):
            str_index = str(index + 1)
            self.params['W' + str_index] = np.sqrt(2.0 / input_size) * np.random.randn(input_size, hidden_size)
            self.params['b' + str_index] = np.zeros(hidden_size)
            self.layers['Affine' + str_index] = Affine(self.params['W' + str_index], self.params['b' + str_index])
            input_size = hidden_size
        str_index = str(len(hidden_size_list) + 1)
        self.params['W' + str_index] = np.sqrt(2.0 / hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b' + str_index] = np.zeros(output_size)
        self.layers['Affine' + str_index] = Affine(self.params['W' + str_index], self.params['b' + str_index])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for index, _ in enumerate(layers):
            grads['W' + str(index + 1)] = self.layers['Affine' + str(index + 1)].dW
            grads['b' + str(index + 1)] = self.layers['Affine' + str(index + 1)].db

        return grads
