import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identify_function(x):
    return x


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

# A = np.array([1, 2, 3, 4])
# print(A)
# print(np.ndim(A))
# print(A.shape)

# B = np.array([[1, 2], [3, 4], [5, 6]])
# C = np.array([7, 8])
# print(np.dot(B, C))

# X = np.array([1.0, 0.5])
# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])
#
# A1 = np.dot(X, W1) + B1
# Z1 = sigmoid(A1)
# print(A1)
# print(Z1)

# W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# B2 = np.array([0.1, 0.2])
#
# A2 = np.dot(Z1, W2) + B2
# Z2 = sigmoid(A2)
# print(A2)
# print(Z2)
#
# W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# B3 = np.array([0.1, 0.2])
#
# A3 = np.dot(Z2, W3) + B3
# Y = identify_function(A3)
# print(A3)
# print(Y)
