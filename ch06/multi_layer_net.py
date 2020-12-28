import numpy as np

from layers import *
from gradient import numerical_gradient
from collections import OrderedDict


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}

        # 가중치 매개변수 초기화 , layer 생성
        self.layers = OrderedDict()
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for index in range(1, len(all_size_list)):
            # HE 초기값(활성화 함수 Relu사용)
            self.params['W' + str(index)] = np.sqrt(2.0 / all_size_list[index - 1]) * np.random.randn(all_size_list[index - 1], all_size_list[index])
            self.params['b' + str(index)] = np.zeros(all_size_list[index])
            self.layers['Affine' + str(index)] = Affine(self.params['W' + str(index)], self.params['b' + str(index)])
            if index <= len(all_size_list) - 2:
                self.layers['Activation_function' + str(index)] = Relu()

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
        for index in range(1, self.hidden_layer_num + 2):
            grads['W' + str(index)] = self.layers['Affine' + str(index)].dW
            grads['b' + str(index)] = self.layers['Affine' + str(index)].db

        return grads
