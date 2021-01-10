import numpy as np
from layers import *
from collections import OrderedDict


class MultiLayerNetExtend:
    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 weight_decay_lambda=0,
                 use_dropout=False,
                 dropout_ration=0.5,
                 use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_size_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 가중치 초기화
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = np.sqrt(2.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        # 계층 생성
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_size_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx - 1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx - 1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                         self.params['beta' + str(idx)])
            self.layers['Activation_function' + str(idx)] = Relu()
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_size_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flag=False):
        y = self.predict(x, train_flag)

        weight_decay = 0
        for idx in range(1, self.hidden_size_num + 1):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flag=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t, train_flag=True)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_size_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_size_num + 1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads
