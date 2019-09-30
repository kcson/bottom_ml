import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6]])
C = np.array([7, 8])
print(np.dot(B, C))
