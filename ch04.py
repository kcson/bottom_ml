import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y1 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# print(np.array(y1))
# print(np.array(y1).reshape(1, 10))
#
# mse = mean_squared_error(np.array(y), np.array(t))
# print(mse)
#
# cee = cross_entropy_error(np.array(y), np.array(t))
# print(cee)
#
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# print(x_train.shape)
# print(t_train.shape)
#
# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# print(batch_mask)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    #print(x.shape)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        # print(tmp_val)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# x = np.arange(0, 20, 0.1)
# y = function_1(x)
# # plt.xlabel("x")
# # plt.ylabel("f(x)")
# # plt.plot(x, y)
#
# y2 = tangent_line(function_1, 10)
# # plt.plot(x, y2(x))
#
# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))
#
# # plt.show()
#
# print(numerical_gradient(function_2, np.array([3., 4.])))
#
# init_x = np.array([-3.0, 4.0])
# gr = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
# print(gr)


def softmax(a):
    if a.ndim == 2:
        a = a.T
        a = a - np.max(a, axis=0)
        y = np.exp(a) / np.sum(np.exp(a), axis=0)
        return y.T

    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        # print(self.W)
        return np.dot(x, self.W)

    def loss(self, x, t):
        # print("=============")
        # print(x)
        # print("=============")

        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


# f = lambda w: net.loss(x, t)
#
# net = simpleNet()
# # print(net.W)
# # print(net.W[0,0])
# x = np.array([0.6, 0.9])
# # p = net.predict(x)
# # print(p)
# # print(np.argmax(p))
# t = np.array([0, 0, 1])
# # print(net.loss(x, t))
#
# dW = numerical_gradient(f, net.W)
# print(dW)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

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


# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
#
# iters_num = 50
# train_size = x_train.shape[0]
# batch_size = 100
# learning_rate = 0.1
#
# network = TwoLayerNet(784, 50, 10)
#
# # 1에폭당 반복 수
# iter_per_epoch = max(train_size / batch_size, 1)
#
# for i in range(iters_num):
#     print(i)
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#
#     grad = network.numerical_gradient(x_batch, t_batch)
#
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
#
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)
#
#     # 1에폭당 정확도 계산
#     # if i % iter_per_epoch == 0:
#     train_acc = network.accuracy(x_train, t_train)
#     test_acc = network.accuracy(x_test, t_test)
#     train_acc_list.append(train_acc)
#     test_acc_list.append(test_acc)
#     print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
#
# # 그래프 그리기
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()
